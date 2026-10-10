args <- commandArgs(trailingOnly=TRUE)
out <- args[1]
if (length(args)>1) .libPaths(c(args[2],.libPaths()))
library(Hmisc)
spec <- read.csv(file.path(out,"fixtures.csv"),stringsAsFactors=FALSE)
orth <- function(x) {
  if (!ncol(x)) return(matrix(0,nrow(x),0))
  s <- svd(x); rank <- sum(s$d > max(dim(x))*.Machine$double.eps*s$d[1])
  if (!rank) return(matrix(0,nrow(x),0))
  s$u[,seq_len(rank),drop=FALSE]
}
wald <- function(y,d,start) {
  q <- ncol(d)-start
  if (!q) return(rep(-Inf,ncol(y)))
  bread <- solve(crossprod(d)); beta <- bread %*% crossprod(d,y)
  e <- y-d%*%beta;h <- rowSums((d%*%bread)*d)
  selected <- seq.int(start+1,ncol(d))
  vapply(seq_len(ncol(y)),function(j) {
    covariance <- bread %*% crossprod(d,d*(e[,j]/(1-h))^2) %*% bread
    as.numeric(crossprod(beta[selected,j],solve(covariance[selected,selected,drop=FALSE],beta[selected,j])))/q
  }, numeric(1))
}
for (row in seq_len(nrow(spec))) {
  name <- spec$name[row]; basis <- spec$basis[row];candidate <- spec$candidate[row]
  d <- read.csv(file.path(out,paste0(name,"-input.csv")))
  u <- as.matrix(read.csv(file.path(out,paste0(name,"-uniforms.csv")),check.names=FALSE))
  x <- as.matrix(d[,grepl("^X",names(d)),drop=FALSE]);raw <- d[,grepl("^Z",names(d)),drop=FALSE];n <- nrow(d)
  groups <- if ("stratum" %in% names(d)) split(seq_len(n),d$stratum) else list(seq_len(n))
  intercepts <- if ("stratum" %in% names(d)) model.matrix(~factor(d$stratum)-1) else matrix(1,n,1)
  q0 <- orth(intercepts);clinical <- as.matrix(raw)
  if (basis=="restricted_cubic_spline") clinical <- do.call(cbind,lapply(raw,function(z) rcspline.eval(z,knots=quantile(z,c(.05,.275,.5,.725,.95),type=7),inclx=TRUE,norm=2)))
  base <- cbind(q0,orth(clinical-q0%*%crossprod(q0,clinical)))
  polynomial <- do.call(cbind,lapply(raw,function(z) {standardized <- (z-mean(z))/sqrt(mean((z-mean(z))^2));cbind(standardized^2,standardized^3)}))
  mean.design <- cbind(base,orth(polynomial-base%*%crossprod(base,polynomial)))
  e <- x-base%*%crossprod(base,x);h <- rowSums(base^2)
  variance <- e^2/(1-h)
  adjusted <- e/sqrt(1-h)
  for (rows in groups) adjusted[rows,] <- sweep(adjusted[rows,,drop=FALSE],2,colMeans(adjusted[rows,,drop=FALSE]))
  mean.fit <- base%*%crossprod(base,e);mean.error <- (e-mean.fit)/(1-h)
  variance.fit <- q0%*%crossprod(q0,variance)
  variance.error <- (variance-variance.fit)/(1-rowSums(q0^2))
  observed <- c(wald(e,mean.design,ncol(base)),wald(variance,base,ncol(q0)))
  maxima <- numeric(nrow(u))
  for (b in seq_len(nrow(u))) {
    if (candidate=="residual_vector") {
      indices <- integer(n)
      for (rows in groups) indices[rows] <- rows[floor(u[b,rows]*length(rows))+1]
      sample <- adjusted[indices,,drop=FALSE]
      ym <- sample-base%*%crossprod(base,sample);yv <- ym^2/(1-h)
    } else {
      signs <- ifelse(u[b,]>=.5,1,-1)
      ym <- mean.fit+mean.error*signs;yv <- variance.fit+variance.error*signs
    }
    maxima[b] <- max(wald(ym,mean.design,ncol(base)),wald(yv,base,ncol(q0)))
  }
  hits <- sum(maxima>=max(observed));B <- length(maxima)
  interval <- c(if(hits==0) 0 else qbeta(.025,hits,B-hits+1),if(hits==B) 1 else qbeta(.975,hits+1,B-hits))
  interval <- (1+B*interval)/(B+1)
  write.csv(data.frame(maximum=maxima),file.path(out,paste0(name,"-r-maxima.csv")),row.names=FALSE)
  write.csv(data.frame(statistic=observed),file.path(out,paste0(name,"-r-observed.csv")),row.names=FALSE)
  write.csv(data.frame(hits=hits,p=(hits+1)/(B+1),lower=interval[1],upper=interval[2],
    mc_uncertain=interval[1]<=.01 && interval[2]>=.01,residual_allowed=(hits+1)/(B+1)>.01 && interval[1]>.01),
    file.path(out,paste0(name,"-r-decision.csv")),row.names=FALSE)
}
capture.output(sessionInfo(),file=file.path(out,"R-session.txt"))

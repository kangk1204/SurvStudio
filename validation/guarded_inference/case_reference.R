# Independent reconstruction from raw external rows and frozen recipe, not Python basis columns.
args <- commandArgs(trailingOnly=TRUE);out <- args[1]
if(length(args)>1) .libPaths(c(args[2],.libPaths()))
suppressPackageStartupMessages({library(survival);library(Hmisc);library(jsonlite)})
recipe <- fromJSON(file.path(out,"recipe.json"),simplifyVector=FALSE)
if(is.null(recipe)) quit(status=0)
encoder <- recipe$clinical$encoder;base <- encoder$base_encoder
records <- list()
emit <- function(cohort,quantity,value) records[[length(records)+1]] <<- data.frame(file=cohort,quantity=quantity,value=as.numeric(value))
for(file in list.files(out,pattern="-private-input.csv$",full.names=TRUE)) {
  name <- sub("-private-input.csv$","",basename(file))
  classes <- rep(NA_character_,length(base$categorical_features));names(classes) <- unlist(base$categorical_features);classes[] <- "character"
  d <- read.csv(file,colClasses=classes,check.names=FALSE,na.strings=c("","NA"));columns <- list()
  for(variable in base$categorical_features) {
    mapping <- base$categorical_mappings[[variable]]
    for(level in mapping$retained_levels) {
      x <- as.numeric(!is.na(d[[variable]]) & d[[variable]]==level)
      columns[[mapping$level_columns[[level]]]] <- x
    }
  }
  for(variable in base$numeric_features) {
    x <- d[[variable]];x[is.na(x)] <- base$numeric_impute_values[[variable]]
    spec <- encoder$spline_specifications[[variable]]
    if(is.null(spec)) columns[[variable]] <- x
    else {
      basis <- Hmisc::rcspline.eval(x,knots=unlist(spec$knots),inclx=TRUE,norm=2)
      basis <- sweep(sweep(basis,2,unlist(spec$means),"-"),2,unlist(spec$scales),"/")
      for(i in seq_len(ncol(basis))) columns[[spec$terms[[i]]]] <- basis[,i]
    }
  }
  for(marker in recipe$markers) {
    median <- recipe$marker_medians[[marker]]
    if(!(marker %in% names(d))) x <- rep(median,nrow(d))
    else {
      x <- d[[marker]]
      if(grepl("-within_cohort$",name)) {
        spread <- sd(x,na.rm=TRUE)
        if(is.na(spread)||spread<=0) x[] <- median
        else x <- recipe$marker_scale[[marker]]$mean+recipe$marker_scale[[marker]]$sd*(x-mean(x,na.rm=TRUE))/spread
      }
      x[is.na(x)] <- median
    }
    columns[[marker]] <- x
  }
  design <- as.matrix(as.data.frame(columns,check.names=FALSE)[,unlist(recipe$model$terms),drop=FALSE])
  risk <- as.vector(design %*% unlist(recipe$model$coefficients))
  expected <- read.csv(file.path(out,paste0(name,"-private-scores.csv")))
  stopifnot(nrow(d)==nrow(expected))
  emit(name,"maximum_prediction_difference",max(abs(risk-expected$linear_predictor)))
  t <- expected$time;e <- expected$event
  emit(name,"C",concordance(Surv(t,e)~risk,reverse=TRUE,timewt="n")$concordance)
  fit <- coxph(Surv(t,e)~risk,ties="efron",control=coxph.control(eps=1e-12,toler.chol=1e-14,iter.max=100))
  emit(name,"calibration_slope",coef(fit)[1])
}
if(length(records)) write.csv(do.call(rbind,records),file.path(out,"independent-R-values.csv"),row.names=FALSE)
aggregate_file <- file.path(out,"external-aggregate.csv")
if(file.exists(aggregate_file)) {
  aggregate <- read.csv(aggregate_file)
  pool <- function(y,se) {
    k <- length(y);w <- 1/se^2;fixed <- sum(w*y)/sum(w);q <- sum(w*(y-fixed)^2)
    tau2 <- max(0,(q-(k-1))/(sum(w)-sum(w^2)/sum(w)));w <- 1/(se^2+tau2)
    mean <- sum(w*y)/sum(w);sd <- sqrt(1/sum(w));scale <- sum(w*(y-mean)^2)/(k-1)
    hk <- qt(.975,k-1)*sd*sqrt(max(1,scale))
    values <- c(estimate=mean,ci_lower=mean-1.96*sd,ci_upper=mean+1.96*sd,tau2=tau2,q=q,k=k,
      hksj_ci_lower=mean-hk,hksj_ci_upper=mean+hk,hksj_scale=scale)
    if(k>=3) {
      spread <- qt(.975,k-2)*sqrt(tau2+sd^2)
      values <- c(values,pi_lower=mean-spread,pi_upper=mean+spread)
    }
    values
  }
  pooled <- list()
  for(endpoint in unique(aggregate$endpoint)) for(scaling in unique(aggregate$scaling)) {
    part <- aggregate[aggregate$endpoint==endpoint & aggregate$scaling==scaling,,drop=FALSE]
    for(quantity in c("c","clinical_c","delta_c","calibration_slope")) {
      lower <- switch(quantity,c="c_lower",clinical_c="clinical_lower",delta_c="delta_lower",calibration_slope="calibration_lower")
      upper <- switch(quantity,c="c_upper",clinical_c="clinical_upper",delta_c="delta_upper",calibration_slope="calibration_upper")
      valid <- complete.cases(part[,c(quantity,lower,upper)]) & part[[upper]]>part[[lower]]
      valid[is.na(valid)] <- FALSE
      if(sum(valid)>=2) {
        values <- pool(part[[quantity]][valid],(part[[upper]][valid]-part[[lower]][valid])/3.92)
        for(field in names(values)) pooled[[length(pooled)+1]] <- data.frame(group=paste(endpoint,scaling,sep=":"),quantity=quantity,field=field,value=values[field])
      }
    }
  }
  if(length(pooled)) write.csv(do.call(rbind,pooled),file.path(out,"independent-R-pooling.csv"),row.names=FALSE)
}
capture.output(sessionInfo(),file=file.path(out,"R-session.txt"))

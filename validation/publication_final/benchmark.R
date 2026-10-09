args<-commandArgs(TRUE)
.libPaths(c(args[3],args[4],.libPaths()))
suppressPackageStartupMessages({library(survival);library(rms)})
d<-read.csv(args[1]);out<-args[2];task<-args[5];engine<-args[6]
x<-data.frame(grade_B=as.numeric(d$grade=='B'),grade_C=as.numeric(d$grade=='C'),z=d$z,binary=d$binary,m0=d$m0)
if(task=='Cox') {
  frame<-cbind(x,time=d$time,event=d$event)
  fit<-function() {
    f<-if(engine=='R_rms') cph(Surv(time,event)~grade_B+grade_C+z+binary+m0,data=frame,method='efron',eps=1e-12,iter.max=100,x=TRUE,y=TRUE) else
       coxph(Surv(time,event)~grade_B+grade_C+z+binary+m0,data=frame,ties='efron',control=coxph.control(eps=1e-12,iter.max=100))
    as.numeric(coef(f))
  }
} else {
  fit<-function() {
    f<-survfit(Surv(d$time,d$event)~d$grade)
    as.numeric(summary(f,times=c(5,10,15),extend=TRUE)$surv)
  }
}
elapsed<-numeric(10)
for(i in seq_len(13)) {start<-proc.time()[['elapsed']];value<-fit();duration<-proc.time()[['elapsed']]-start;if(i>3) elapsed[i-3]<-duration}
write.csv(data.frame(run=1:10,seconds=elapsed),out,row.names=FALSE)
write.csv(data.frame(value=value),paste0(out,'.values.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=paste0(out,'.session.txt'))

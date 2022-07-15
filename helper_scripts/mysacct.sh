export SLURM_TIME_FORMAT=relative
FMTSTR=JobId,State,Start,Elapsed,End
echo $FMTSTR
sacct -u todorov1 --format $FMTSTR | sed -n '1~2p' | tail "-$1"
#sacct -u todorov1 --format JobId,State,Elapsed,Start,End,AveRss,MaxVMSize,MaxVMSizeTask,ReqMem | sed -n '1~2p' 

#sacct -u todorov1 --format JobId,State,Submit,Start,End
#Submit,
#sed -n 1~2 -- for every second line
#Specify  the  format  used  to report time stamps. A value of standard, the
#default value, generates output in the form
#"year-month-dateThour:minute:second".   A  value  of  relative returns only
#"hour:minute:second" if the current day.  For other dates  in  the  current
#year  it prints the "hour:minute" preceded by "Tomorr" (tomorrow), "Ystday"
#(yesterday), the name of the day for the coming week  (e.g.  "Mon",  "Tue",
#etc.),  otherwise  the  date (e.g. "25 Apr").  For other years it returns a
#date month and year without a time (e.g.  "6 Jun 2012"). All  of  the  time
# stamps use a 24 hour format.
#
# A  valid  strftime()  format can also be specified. For example, a value of
# "%a %T" will report the day of  the  week  and  a  time  stamp  (e.g.  "Mon
# 12:34:56").
#

# just to check when I have submitted. I need to set time
FMTSTR=JobID,State,Submit,Start
export SLURM_TIME_FORMAT=relative
echo $FMTSTR
#month/day-hour/minute
sacct -S 11/15-00:40 --format $FMTSTR | sed -n '1~2p' | tail "-$1" 

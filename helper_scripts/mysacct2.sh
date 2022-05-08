#!/usr/bin/bash
# just to check when I have submitted. I need to set time
FMTSTR=JobID,State,Submit,Start,Elapsed
export SLURM_TIME_FORMAT=relative
echo $FMTSTR
#month/day-hour/minute
sacct -S 04/20-00:00 --format $FMTSTR | sed -n '1~2p' | tail "-$1" 

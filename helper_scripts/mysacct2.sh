#!/usr/bin/bash
# just to check when I have submitted. I need to set time
FMTSTR=JobID%-11,State%7,Submit%5,Start%5,Elapsed
export SLURM_TIME_FORMAT=relative
echo $FMTSTR
#month/day-hour/minute
sacct -S2022-06-18-00:00 --format $FMTSTR | sed -n '1~2p' | tail "-$1" 

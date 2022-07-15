#!/usr/bin/bash
slurmout_path=$CODE/slurmout
###!/usr/bin/python3
#if [ $# -lt 1 ]; then
#  echo "Need 1 argument -- index of job"
#  echo "or two arguments -- jobId and index of job"
#  exit 1
#fi

if [ $# -eq 0 ]; then
  lastfilestr=`ls -ltr $slurmout_path/*.out  | grep -v ^d | tail -1`
elif [ $# -eq 1 ]; then
  lastfilestr=`ls -ltr $slurmout_path/*_*_$1.out  | grep -v ^d | tail -1`
elif [ $# -eq 2 ]; then
  lastfilestr=`ls -ltr $slurmout_path/*_$1_$2.out | grep -v ^d | tail -1`
fi

#echo "$lastfilestr"
#lastfile=`echo "$lastfilestr" | tail -1`

lastfile=`echo $lastfilestr | awk '{ print $NF }'`
#lastfile=slurmout/"$lastfile"
lastfile="$lastfile"

#lastfile=`find slurmout -maxdepth 1 -not -type d | tail -1`
echo ":::Showing end of lastfile=$lastfile"
echo ""
vim -R ${lastfile[0]} 

#!/bin/bash
CR=$OSCBAGDIS_DATAPROC_CODE/run
#RUNSTRINGS_FN="$CR/_runstrings_ML.txt"
RUNSTRINGS_FN="$CR/_runstrings_genfeats.txt"
EFF_ID=0
ID=0
#while [ $NUMRS -gt $SHIFT_ID ]; do
if [ $# -ge 1 ]; then
  N=$1
  echo N=$N
else
  N=`wc -l $RUNSTRINGS_FN`
  N=`echo $N | cut -d ' ' -f1`
  echo N=$N
fi

mapfile -t RUNSTRINGS < $RUNSTRINGS_FN

#Nm1=$(( $N - 1 ))
Nm1=$[N - 1]
echo Nm1=$Nm1
for (( i=0; i<=Nm1; i++ )); do
  SLURM_ARRAY_JOB_ID=$i
  RUNSTRING_CUR=${RUNSTRINGS[$i]}
  echo "-------- Starting runstring # $i"
  echo $RUNSTRING_CUR

  #py=ipython
  py=python
  export PYTHONPATH=$OSCBAGDIS_DATAPROC_CODE:$PYTHONPATH
  $py $CR/$RUNSTRING_CUR --SLURM_job_id "$JOBID"_"$JOB_ARRAY_IND" --runstring_ind $i 
  EC=$?
  if [ $EC -gt 0 ]; then
    echo "EC=$EC"
    exit $EC
  fi
  echo "-------- Finished runstring # $i"
  echo $RUNSTRING_CUR
  echo "-------------------------------"
done

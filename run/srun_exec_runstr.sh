#!/bin/bash
RUNSTRINGS_FN=$1
JOBID=$2
RUNSTRING_IND=$3
mapfile -t RUNSTRINGS < $RUNSTRINGS_FN
RUNSTRING_CUR=${RUNSTRINGS[$RUNSTRING_IND]} 
R=$(eval echo $RUNSTRING_CUR)
#python $R
echo ----------
echo    Runstring ind = $RUNSTRING_IND is
echo $R
#py=ipython
py=python
export PYTHONPATH=$OSCBAGDIS_DATAPROC_CODE:$PYTHONPATH
$py -c "import os; print('cwd for $py=',os.getcwd() );"
$py $CODE/run/$R --SLURM_job_id "$JOBID"_"$RUNSTRING_IND" --calc_MI 0

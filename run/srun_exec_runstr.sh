#!/bin/bash
RUNSTRINGS_FN="$CODE/run/_runstrings.txt"
JOBID=$1
RUNSTRING_IND=$2
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
$py $CODE/run/$R --SLURM_job_id "$JOBID"_"$RUNSTRING_IND"

#!/bin/bash
# this tells to exit if any error occurs (by def bash does not do it)
set -e
RUNSTRINGS_FN=$1
JOBID=$2
RUNSTRING_IND=$3
mapfile -t RUNSTRINGS < $RUNSTRINGS_FN
RUNSTRING_CUR=${RUNSTRINGS[$RUNSTRING_IND]}
echo ----------
echo    Runstring ind = $RUNSTRING_IND is
#R=$(eval echo $RUNSTRING_CUR)
#echo "$R"
echo $RUNSTRING_CUR

#py=ipython
py=python
export PYTHONPATH=$OSCBAGDIS_DATAPROC_CODE:$PYTHONPATH
$py -c "import os; print('cwd for $py=',os.getcwd() );"
#$py $CODE/run/$R --SLURM_job_id "$JOBID"_"$RUNSTRING_IND" --calc_MI 0
$py $CODE/run/$RUNSTRING_CUR --SLURM_job_id "$JOBID"_"$RUNSTRING_IND" --calc_MI 0

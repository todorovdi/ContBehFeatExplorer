#!/bin/bash
# this tells to exit if any error occurs (by def bash does not do it)
date
set -e
RUNSTRINGS_FN=$1
JOBID=$2
RUNSTRING_IND=$3
JOB_ARRAY_IND=$4
mapfile -t RUNSTRINGS < $RUNSTRINGS_FN
echo ----------
echo    "Runstring ind = $RUNSTRING_IND"
#R=$(eval echo $RUNSTRING_CUR)
#echo "$R"
#echo $RUNSTRING_CUR

#num_subj=6
#num_rs=${#RUNSTRINGS[@]}
#echo num_rs=$num_rs
##shft_=$(expr $RUNSTRING_IND/$num_subj)
##let shft=$RUNSTRING_IND/$num_subj
##shft=$((RUNSTRING_IND/num_subj))
##echo "ffff-0, $num_subj,$RUNSTRING_IND"
## for some weird reason all bash-based arithmetics 
## options give bugs when something is zero 
#shft_=`python -c "print($RUNSTRING_IND // $num_subj)"`
#echo shft_=$shft_
#num_pref_per_subj=`python -c "print($num_rs // $num_subj)"`
#echo "num_pref_per_subj = $num_pref_per_subj"
#subj_ind=`python -c "print($RUNSTRING_IND % $num_subj)"`
##echo "ffff1"
#baseind=`python -c  "print($subj_ind * $num_pref_per_subj)"`
##echo "ffff2"
#newind=`python -c  "print( $shft_ + $baseind)"`


# this way we run first all first prefixes, then all second and so on
#RSID=$newind
RSID=$RUNSTRING_IND
echo "final RSID is $RSID"

RUNSTRING_CUR=${RUNSTRINGS[$RSID]}
echo $RUNSTRING_CUR

#py=ipython
#py=python
py=$python_correct_ver
export PYTHONPATH=$OSCBAGDIS_DATAPROC_CODE:$PYTHONPATH
$py -c "import os; print('cwd for $py=',os.getcwd() );"
echo "srun_exrc_runstr0 which python"
which $py


echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES , PMI_RANK=$PMI_RANK"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  $py -c "import GPUtil; print('GPUs = ',GPUtil.getAvailable()); import pycuda.driver as cuda_driver; cuda_driver.init(); "
fi

$RUNSTRING_CUR --SLURM_job_id "$JOBID"_"$JOB_ARRAY_IND" --runstring_ind $RUNSTRING_IND 
exit $?

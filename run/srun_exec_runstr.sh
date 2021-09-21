#!/bin/bash
# this tells to exit if any error occurs (by def bash does not do it)
set -e
RUNSTRINGS_FN=$1
JOBID=$2
RUNSTRING_IND=$3
mapfile -t RUNSTRINGS < $RUNSTRINGS_FN
echo ----------
echo    "Runstring ind = $RUNSTRING_IND"
#R=$(eval echo $RUNSTRING_CUR)
#echo "$R"
#echo $RUNSTRING_CUR

num_subj=6
num_rs=${#RUNSTRINGS[@]}
echo num_rs=$num_rs
#shft_=$(expr $RUNSTRING_IND/$num_subj)
#let shft=$RUNSTRING_IND/$num_subj
#shft=$((RUNSTRING_IND/num_subj))
#echo "ffff-0, $num_subj,$RUNSTRING_IND"
# for some weird reason all bash-based arithmetics 
# options give bugs when something is zero 
shft_=`python -c "print($RUNSTRING_IND // $num_subj)"`
echo shft_=$shft_
num_pref_per_subj=`python -c "print($num_rs // $num_subj)"`
echo "num_pref_per_subj = $num_pref_per_subj"
subj_ind=`python -c "print($RUNSTRING_IND % $num_subj)"`
#echo "ffff1"
baseind=`python -c  "print($subj_ind * $num_pref_per_subj)"`
#echo "ffff2"
newind=`python -c  "print( $shft_ + $baseind)"`


# this way we run first all first prefixes, then all second and so on
RSID=$newind
echo "modified RUNSTRING_IND is $RSID"
#RSID=$RUNSTRING_IND

RUNSTRING_CUR=${RUNSTRINGS[$RSID]}
echo $RUNSTRING_CUR

#py=ipython
py=python
export PYTHONPATH=$OSCBAGDIS_DATAPROC_CODE:$PYTHONPATH
$py -c "import os; print('cwd for $py=',os.getcwd() );"
$py $CODE/run/$RUNSTRING_CUR --SLURM_job_id "$JOBID"_"$RSID" --calc_MI 0

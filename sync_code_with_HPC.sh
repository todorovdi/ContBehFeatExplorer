#!/bin/bash
#rsync -avz --progress -e ssh $HOME/osccode/data_proc/* jusuf:data_proc_code/
#JUSUF_CODE="jusuf:data_proc_code"
if [[ $# -lt 1 ]]; then
  echo "Need at least one arg"
  exit 1
fi

SYNC_MODE="both"
if [[ $# -eq 2 ]]; then
  SYNC_MODE=$2
fi
#FETCH_NPZ_AND_JSON=1
FETCH_NPZ_AND_JSON=0

if [[ $SYNC_MODE != "get_from" && $SYNC_MODE != "send_to" && $SYNC_MODE != "both"  ]]; then
  echo "wrong param, exit"
  exit 1
fi

run="python3 _rsync_careful.py"
#run="ipython3 -i _rsync_careful.py --"
  
echo '' > sync_dest_changes.log
 
RUNTYPE=$1
DRY_RUN_FLAG=""
if [[ $1 == "dry" ]]; then
  echo " ----------  DRY RUN  ------------"
  DRY_RUN_FLAG="n"
elif [[ $1 == "normal" ]]; then
  echo " ----------  NORMAL RUN  ------------"
else
  echo " ----------  WRONG CMD OPTION  ------------"
  exit 1
fi

LOCAL_DIR="/home/demitau/osccode/data_proc"
#FLAGS="-rtvh$DRY_RUN_FLAG --progress"  # removed z
# for dry run
#FLAGS="-rtvzn --progress"; echo "DRY RUN!!"

DIRECT_SSH=0
. remount_HPC.sh

  
#$run --mode:$RUNTYPE --exclude="*HPC*.py"  "$LOCAL_DIR/*.py"  "$JUSUF_CODE/"
#exit 1

#FNS=`ls -L *.{py,sh}`
#rsync $FLAGS --progress $SSH_FLAG $FNS jusuf:data_proc_code/
subdir=run
if [[ $SYNC_MODE != "get_from" ]]; then
  echo "  sync souce code"
  $run --mode:$RUNTYPE --exclude="*HPC*.py" --exclude="_rsync*.py" "$LOCAL_DIR/*.py"  "$JUSUF_CODE/"
  $run --mode:$RUNTYPE --exclude="sync*HPC.sh"  "$LOCAL_DIR/*.sh"  "$JUSUF_CODE/"
  #$run --mode:$RUNTYPE "$LOCAL_DIR/*.m"  "$JUSUF_CODE/"  # we don't need *.m files on HPC
  $SLEEP
  echo "  sync run files (excluding sh, only py)"
  #rsync $FLAGS $SSH_FLAG --exclude="*HPC.sh" --exclude="sbatch*" --exclude=srun_pipeline.sh --exclude=srun_exec_runstr.sh $LOCAL_DIR/$subdir/*.{py,sh}  $JUSUF_CODE/$subdir/
  $run --mode:$RUNTYPE --exclude="indtool.py" --exclude="_utils_indtool.py" "$LOCAL_DIR/$subdir/*.py"   "$JUSUF_CODE/$subdir/"
  #echo "  sync matlab_compiled"
  #$run --mode:$RUNTYPE "$LOCAL_DIR/matlab_compiled/" "$JUSUF_CODE/matlab_compiled" 
  $SLEEP
  $run --mode:$RUNTYPE "$LOCAL_DIR/subj_corresp.json"  "$JUSUF_CODE/"
fi
# if only send we don't want to receive
if [[ $SYNC_MODE != "send_to" ]]; then
  echo "  rev sync run files (excluding sh, only py)"
  $run --mode:$RUNTYPE "$JUSUF_CODE/$subdir/*.sh" "$LOCAL_DIR/$subdir/"
  #$run --mode:$RUNTYPE "$JUSUF_CODE/$subdir/{indtool,_utils_indtool}.py" "$LOCAL_DIR/$subdir/"
  $run --mode:$RUNTYPE "$JUSUF_CODE/$subdir/_subrun*HPC.py" "$LOCAL_DIR/$subdir/"
  $SLEEP
  echo "  rev sync tmux"
  $run --mode:$RUNTYPE "$JUSUF_CODE/.tmux*.conf" "$LOCAL_DIR"
  echo "  rev sync souce code"
  $run --mode:$RUNTYPE --exclude="sync*HPC.sh" "$JUSUF_CODE/*HPC.py" "$LOCAL_DIR/"
fi


# we don't copy HPC back
if [[ $SYNC_MODE != "get_from" ]]; then
  $SLEEP
  echo "  sync json"
  $run --mode:$RUNTYPE --exclude="*HPC" "$LOCAL_DIR/*.json"  "$JUSUF_CODE/"
  $SLEEP
  # send params to HPC except those ending with HPC
  echo "  sync params"  # I want to care about *HPC_fast too
  $run --mode:$RUNTYPE --exclude="*HPC*.ini" "$LOCAL_DIR/params/*.ini" "$JUSUF_CODE/params/"
  $SLEEP
  echo "  sync test data"
  $run --mode:$RUNTYPE "$LOCAL_DIR/test_data/*.py" "$JUSUF_CODE/test_data/"
fi


if [[ $SYNC_MODE != "send_to" ]]; then
  echo "  rev sync params"
  # receive HPC params back here
  $run --mode:$RUNTYPE "$JUSUF_CODE/params/*HPC*.ini" "$LOCAL_DIR/params/" 
  echo "  rev sync helper_scripts"
  $run --mode:$RUNTYPE "$JUSUF_CODE/helper_scripts/*.sh" "$LOCAL_DIR/helper_scripts/" 

fi



if [[ $SYNC_MODE != "get_from" ]]; then
  # save current code version and make it transferable to HPC (we don't have git there)
  v=`git tag | tail -1` 
  h=`git rev-parse --short HEAD`
  echo "$v, hash=$h" > last_code_ver_synced_with_HPC.txt
  wait

  echo "  sync req and code ver"
  $run --mode:$RUNTYPE "$LOCAL_DIR/requirements.txt" "$JUSUF_CODE/"
  #$SLEEP
  $run --mode:$RUNTYPE "$LOCAL_DIR/last_code_ver_synced_with_HPC.txt" "$JUSUF_CODE/"
fi



if [[ $SYNC_MODE != "send_to" ]]; then
  if [[ $FETCH_NPZ_AND_JSON -ne 0 ]]; then
    echo "  rev sync EXPORT"
    rsync -rtvz --progress $JUSUF_BASE/data_duss/per_subj_per_medcond_best_LFP_wholectx/*.pkl $DATA_DUSS/per_subj_per_medcond_best_LFP_wholectx/

    sd=joint_noskip
    mkdir $DATA_DUSS/$sd  
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/EXPORT*.npz" "$DATA_DUSS/$sd" 
    sd=per_subj_per_medcond_best_LFP
    mkdir $DATA_DUSS/$sd  
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/EXPORT*.npz" "$DATA_DUSS/$sd" 
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/beh_states_durations.npz" "$DATA_DUSS/$sd" 
    sd=per_subj_per_medcond_best_LFP_wholectx
    mkdir $DATA_DUSS/$sd  
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/EXPORT*.npz" "$DATA_DUSS/$sd" 
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/perftable*.npz" "$DATA_DUSS/$sd" 
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/beh_states_durations.npz" "$DATA_DUSS/$sd" 
    echo "  rev sync bestLFP"
    sd=searchLFP_both_sides
    mkdir $DATA_DUSS/$sd  
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides.json" "$DATA_DUSS/$sd" 
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides_ext.json" "$DATA_DUSS/$sd" 
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides.json" "$LOCAL_DIR"
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides_ext.json" "$LOCAL_DIR"
    sd=searchLFP_both_sides_oversample
    $run --mode:$RUNTYPE "$JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides_ext.json" "$LOCAL_DIR/best_LFP_info_both_sides_oversample.json"
  fi
fi


cat sync_dest_changes.log

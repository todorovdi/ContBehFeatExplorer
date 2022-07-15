#!/bin/bash
#rsync -avz --progress -e ssh $HOME/osccode/data_proc/* jusuf:data_proc_code/
#JUSUF="jusuf:data_proc_code"
if [[ $# -lt 1 ]]; then
  echo "Need at least one arg"
  exit 1
fi

SYNC_MODE="both"
if [[ $# -eq 2 ]]; then
  SYNC_MODE=$2
fi

if [[ $SYNC_MODE != "get_from" && $SYNC_MODE != "send_to" && $SYNC_MODE != "both"  ]]; then
  echo "wrong param, exit"
  exit 1
fi

  
 
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
FLAGS="-rtvh$DRY_RUN_FLAG --progress"  # removed z
# for dry run
#FLAGS="-rtvzn --progress"; echo "DRY RUN!!"


DIRECT_SSH=0
if [ $DIRECT_SSH -ne 0 ]; then
  echo "Using rsync -e ssh"
  SSH_FLAG="-e ssh"
  JUSUF="judac:data_proc_code"
  JUSUF_BASE="judac:ju_oscbagdis"
  #SLEEP="sleep 1s"
  SLEEP=""
else
  mountpath="$HOME/ju_oscbagdis"
  numfiles=`ls $mountpath | wc -l`
  MQR=`mountpoint -q "$mountpath"`
  while [ $numfiles -eq 0 ] || ! mountpoint -q "$mountpath"; do
    echo "not mounted! trying to remount; numfiles=$numfiles MQR=$MQR"
    sudo umount -l $mountpath # would not work if I run on cron
    sshfs judac:/p/project/icei-hbp-2020-0012/OSCBAGDIS $mountpath
    #exit 1
    sleep 3s
    numfiles=`ls $mountpath | wc -l`
    MQR=`mountpoint -q "$mountpath"`
  done

  echo "Using mounted sshfs"
  SSH_FLAG=""
  JUSUF="$HOME/ju_oscbagdis/data_proc_code"
  JUSUF_BASE="$HOME/ju_oscbagdis"
  SLEEP=""
fi

#FNS=`ls -L *.{py,sh}`
#rsync $FLAGS --progress $SSH_FLAG $FNS jusuf:data_proc_code/
subdir=run
if [[ $SYNC_MODE != "get_from" ]]; then
  echo "  sync souce code"
  rsync $FLAGS $SSH_FLAG --exclude="*HPC.py" --exclude="sync*HPC.sh"  $LOCAL_DIR/*.{py,sh,m}  $JUSUF/
  $SLEEP
  echo "  sync run files (excluding sh, only py)"
  #rsync $FLAGS $SSH_FLAG --exclude="*HPC.sh" --exclude="sbatch*" --exclude=srun_pipeline.sh --exclude=srun_exec_runstr.sh $LOCAL_DIR/$subdir/*.{py,sh}  $JUSUF/$subdir/
  rsync $FLAGS $SSH_FLAG $LOCAL_DIR/$subdir/*.py --exclude=indtool.py --exclude=_utils_indtool.py  $JUSUF/$subdir/
  echo "  sync matlab_compiled"
  rsync $FLAGS $SSH_FLAG $LOCAL_DIR/matlab_compiled/  $JUSUF/matlab_compiled 
  $SLEEP
fi
# if only send we don't want to receive
if [[ $SYNC_MODE != "send_to" ]]; then
  echo "  rev sync run files (excluding sh, only py)"
  rsync $FLAGS $SSH_FLAG  $JUSUF/$subdir/*.sh  $LOCAL_DIR/$subdir/
  rsync $FLAGS $SSH_FLAG  $JUSUF/$subdir/{indtool,_utils_indtool}.py  $LOCAL_DIR/$subdir/
  rsync $FLAGS $SSH_FLAG  $JUSUF/$subdir/_subrun*HPC.py  $LOCAL_DIR/$subdir/
  $SLEEP
  echo "  rev sync tmux"
  rsync $FLAGS $SSH_FLAG $JUSUF/.tmux*.conf  $LOCAL_DIR
  echo "  rev sync souce code"
  rsync $FLAGS $SSH_FLAG --exclude="sync*HPC.sh" $JUSUF/*HPC.py $LOCAL_DIR/
fi


if [[ $SYNC_MODE != "get_from" ]]; then
  $SLEEP
  echo "  sync json"
  rsync $FLAGS $SSH_FLAG --exclude="*HPC" $LOCAL_DIR/*.json  $JUSUF/
  $SLEEP
  echo "  sync params"  # I want to care about *HPC_fast too
  rsync $FLAGS $SSH_FLAG --exclude="*HPC*.ini" $LOCAL_DIR/params/*.ini $JUSUF/params/
  $SLEEP
  echo "  sync test data"
  rsync $FLAGS $SSH_FLAG $LOCAL_DIR/test_data/*.py $JUSUF/test_data/
fi

if [[ $SYNC_MODE != "send_to" ]]; then
  echo "  rev sync params"
  rsync $FLAGS $SSH_FLAG $JUSUF/params/*HPC*.ini $LOCAL_DIR/params/ 
  echo "  rev sync helper_scripts"
  rsync $FLAGS $SSH_FLAG $JUSUF/helper_scripts/*.sh $LOCAL_DIR/helper_scripts/ 
  echo "  rev sync EXPORT"
  sd=joint_noskip
  mkdir $DATA_DUSS/$sd  
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/EXPORT*.npz $DATA_DUSS/$sd 
  sd=per_subj_per_medcond_best_LFP
  mkdir $DATA_DUSS/$sd  
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/EXPORT*.npz $DATA_DUSS/$sd 
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/beh_states_durations.npz $DATA_DUSS/$sd 
  sd=per_subj_per_medcond_best_LFP_wholectx
  mkdir $DATA_DUSS/$sd  
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/EXPORT*.npz $DATA_DUSS/$sd 
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/perftable*.npz $DATA_DUSS/$sd 
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/beh_states_durations.npz $DATA_DUSS/$sd 
  echo "  rev sync bestLFP"
  sd=searchLFP_both_sides
  mkdir $DATA_DUSS/$sd  
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides.json $DATA_DUSS/$sd 
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides_ext.json $DATA_DUSS/$sd 
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides.json $LOCAL_DIR
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides_ext.json $LOCAL_DIR
  sd=searchLFP_both_sides_oversample
  rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides_ext.json $LOCAL_DIR/best_LFP_info_both_sides_oversample.json
fi


if [[ $SYNC_MODE != "get_from" ]]; then
  # save current code version and make it transferable to HPC (we don't have git there)
  git tag | tail -1 > last_code_ver_synced_with_HPC.txt

  echo "  sync req"
  rsync $FLAGS $SSH_FLAG "$LOCAL_DIR/requirements.txt" $JUSUF/
  rsync $FLAGS $SSH_FLAG "$LOCAL_DIR/last_code_ver_synced_with_HPC.txt" $JUSUF/
  $SLEEP
fi

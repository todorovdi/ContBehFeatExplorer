#!/bin/bash
#rsync -avz --progress -e ssh $HOME/osccode/data_proc/* jusuf:data_proc_code/
#JUSUF="jusuf:data_proc_code"
if [[ $# -ne 1 ]]; then
  echo "Need one arg"
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

ZBOOK_DIR="/home/demitau/osccode/data_proc"
FLAGS="-rtvz$DRY_RUN_FLAG --progress"
# for dry run
#FLAGS="-rtvzn --progress"; echo "DRY RUN!!"


DIRECT_SSH=0
if [ $DIRECT_SSH -ne 0 ]; then
  echo "Using rsync -e ssh"
  SSH_FLAG="-e ssh"
  JUSUF="judac:data_proc_code"
  JUSUF_BASE="judac:ju_oscbagdis"
  SLEEP="sleep 1s"
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
echo "  rsync souce code"
rsync $FLAGS $SSH_FLAG --exclude="*HPC.py" --exclude="sync*HPC.sh"  $ZBOOK_DIR/*.{py,sh,m}  $JUSUF/
$SLEEP
subdir=run
echo "  rsync run files (excluding sh, only py)"
#rsync $FLAGS $SSH_FLAG --exclude="*HPC.sh" --exclude="sbatch*" --exclude=srun_pipeline.sh --exclude=srun_exec_runstr.sh $ZBOOK_DIR/$subdir/*.{py,sh}  $JUSUF/$subdir/
rsync $FLAGS $SSH_FLAG $ZBOOK_DIR/$subdir/*.py --exclude=indtool.py --exclude=_utils_indtool.py  $JUSUF/$subdir/
echo "  rsync matlab_compiled"
rsync $FLAGS $SSH_FLAG $ZBOOK_DIR/matlab_compiled/  $JUSUF/matlab_compiled 
$SLEEP
echo "  rev rsync run files (excluding sh, only py)"
rsync $FLAGS $SSH_FLAG  $JUSUF/$subdir/*.sh  $ZBOOK_DIR/$subdir/
rsync $FLAGS $SSH_FLAG  $JUSUF/$subdir/{indtool,_utils_indtool}.py  $ZBOOK_DIR/$subdir/
rsync $FLAGS $SSH_FLAG  $JUSUF/$subdir/_subrun*HPC.py  $ZBOOK_DIR/$subdir/
$SLEEP
echo "  rev rsync souce code"
rsync $FLAGS $SSH_FLAG --exclude="sync*HPC.sh" $JUSUF/*HPC.py $ZBOOK_DIR/


$SLEEP
echo "  rsync json"
rsync $FLAGS $SSH_FLAG --exclude="*HPC" $ZBOOK_DIR/*.json  $JUSUF/
$SLEEP
echo "  rsync params"
rsync $FLAGS $SSH_FLAG --exclude="*HPC*.ini" $ZBOOK_DIR/params/*.ini $JUSUF/params/
echo "  rev rsync params"
rsync $FLAGS $SSH_FLAG $JUSUF/params/*HPC*.ini $ZBOOK_DIR/params/ 
echo "  rsync test data"
rsync $FLAGS $SSH_FLAG $ZBOOK_DIR/test_data/*.py $JUSUF/test_data/
echo "  rev rsync helper_scripts"
rsync $FLAGS $SSH_FLAG $JUSUF/helper_scripts/*.sh $ZBOOK_DIR/helper_scripts/ 
echo "  rev rsync EXPORT"
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
echo "  rev rsync bestLFP"
sd=searchLFP_both_sides
mkdir $DATA_DUSS/$sd  
rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides.json $DATA_DUSS/$sd 
rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides_ext.json $DATA_DUSS/$sd 
rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides.json $ZBOOK_DIR
rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides_ext.json $ZBOOK_DIR
sd=searchLFP_both_sides_oversample
rsync $FLAGS $SSH_FLAG $JUSUF_BASE/data_duss/$sd/best_LFP_info_both_sides_ext.json $ZBOOK_DIR/best_LFP_info_both_sides_oversample.json


# save current code version and make it transferable to HPC (we don't have git there)
git tag | tail -1 > last_code_ver_synced_with_HPC.txt

echo "  rsync req"
rsync $FLAGS $SSH_FLAG "$ZBOOK_DIR/requirements.txt" $JUSUF/
rsync $FLAGS $SSH_FLAG "$ZBOOK_DIR/last_code_ver_synced_with_HPC.txt" $JUSUF/
$SLEEP

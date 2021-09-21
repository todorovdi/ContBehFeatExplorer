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
  SLEEP="sleep 1s"
else
  mountpath="$HOME/ju_oscbagdis"
  numfiles=`ls $mountpath | wc -l`
  if [ $numfiles -eq 0 ]; then
    echo "not mounted! trying to remount"
    sudo umount -l $mountpath # would not work if I run on cron
    sshfs judac:/p/project/icei-hbp-2020-0012/OSCBAGDIS $mountpath
    #exit 1
  fi

  echo "Using mounted sshfs"
  SSH_FLAG=""
  JUSUF="$HOME/ju_oscbagdis/data_proc_code"
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
rsync $FLAGS $SSH_FLAG $ZBOOK_DIR/$subdir/*.py --exclude=indtool.py  $JUSUF/$subdir/
$SLEEP
echo "  rev rsync run files (excluding sh, only py)"
rsync $FLAGS $SSH_FLAG  $JUSUF/$subdir/*.sh  $ZBOOK_DIR/$subdir/
rsync $FLAGS $SSH_FLAG  $JUSUF/$subdir/indtool.py  $ZBOOK_DIR/$subdir/
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


# save current code version and make it transferable to HPC (we don't have git there)
git tag | tail -1 > last_code_ver_synced_with_HPC.txt

echo "  rsync req"
rsync $FLAGS $SSH_FLAG "$ZBOOK_DIR/requirements.txt" $JUSUF/
rsync $FLAGS $SSH_FLAG "$ZBOOK_DIR/last_code_ver_synced_with_HPC.txt" $JUSUF/
$SLEEP

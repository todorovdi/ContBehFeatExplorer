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

echo "  rsync req"
rsync $FLAGS $SSH_FLAG "$ZBOOK_DIR/requirements.txt" $JUSUF/
$SLEEP
#FNS=`ls -L *.{py,sh}`
#rsync $FLAGS --progress $SSH_FLAG $FNS jusuf:data_proc_code/
echo "  rsync souce code"
rsync $FLAGS $SSH_FLAG --exclude="*HPC.py" --exclude="sync*HPC.sh"  $ZBOOK_DIR/*.{py,sh,m}  $JUSUF/

echo "  rev rsync souce code"
rsync $FLAGS $SSH_FLAG --exclude="sync*HPC.sh" $JUSUF/ $ZBOOK_DIR/*HPC.py
$SLEEP
echo "  rsync json"
rsync $FLAGS $SSH_FLAG --exclude="*HPC" $ZBOOK_DIR/*.json  $JUSUF/
$SLEEP
subdir=run
echo "  rsync run files (excluding sh, only py)"
#rsync $FLAGS $SSH_FLAG --exclude="*HPC.sh" --exclude="sbatch*" --exclude=srun_pipeline.sh --exclude=srun_exec_runstr.sh $ZBOOK_DIR/$subdir/*.{py,sh}  $JUSUF/$subdir/
rsync $FLAGS $SSH_FLAG $ZBOOK_DIR/$subdir/*.py  $JUSUF/$subdir/
$SLEEP
echo "  rev rsync run files (excluding sh, only py)"
rsync $FLAGS $SSH_FLAG  $JUSUF/$subdir/*.sh  $ZBOOK_DIR/$subdir/
$SLEEP
echo "  rsync params"
rsync $FLAGS $SSH_FLAG --exclude="*HPC*.ini" $ZBOOK_DIR/params/*.ini $JUSUF/params/
echo "  rev rsync params"
rsync $FLAGS $SSH_FLAG $JUSUF/params/*HPC*.ini $ZBOOK_DIR/params/ 
echo "  rsync test data"
rsync $FLAGS $SSH_FLAG $ZBOOK_DIR/test_data/*.py $JUSUF/test_data/

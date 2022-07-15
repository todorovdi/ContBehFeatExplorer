#!/bin/bash
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

LOCAL=1
REMOTE=1

if [ $LOCAL -gt 0 ]; then
  echo " self clean jupyter"
  sd=jupyter_debug
  mkdir $ZBOOK_DIR/"$sd"_cleaned  
  python -m nbconvert --ClearOutputPreprocessor.enabled=True $ZBOOK_DIR/$sd/*.ipynb --to notebook --output-dir=$ZBOOK_DIR/"$sd"_cleaned   

  ##
  sd=jupyter_release
  mkdir $ZBOOK_DIR/"$sd"_cleaned  
  python -m nbconvert --ClearOutputPreprocessor.enabled=True $ZBOOK_DIR/$sd/*.ipynb --to notebook --output-dir=$ZBOOK_DIR/"$sd"_cleaned  
fi


if [ $REMOTE -gt 0 ]; then
  echo "  rev rsync jupyter debug"
  sd=jupyter_debug
  sd_HPC="$sd"_cleaned
  sd_loc="$sd"_HPC_cleaned
  mkdir $ZBOOK_DIR/$sd_loc  
  mkdir $JUSUF/$sd_HPC
  python -m nbconvert --ClearOutputPreprocessor.enabled=True $JUSUF/$sd/*.ipynb --to notebook --output-dir=$JUSUF/$sd_HPC
  rsync $FLAGS $SSH_FLAG $JUSUF/$sd_HPC/*.ipynb $ZBOOK_DIR/$sd_loc  

  echo "  rev rsync jupyter release"
  sd=jupyter_release
  sd_HPC="$sd"_cleaned
  sd_loc="$sd"_HPC_cleaned
  mkdir $ZBOOK_DIR/$sd_loc  
  mkdir $JUSUF/$sd_HPC
  python -m nbconvert --ClearOutputPreprocessor.enabled=True $JUSUF/$sd/*.ipynb --to notebook --output-dir=$JUSUF/$sd_HPC
  rsync $FLAGS $SSH_FLAG $JUSUF/$sd_HPC/*.ipynb $ZBOOK_DIR/$sd_loc
fi

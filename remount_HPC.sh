#!/bin/bash
# REQURES DIRECT_SSH to be defined, normally this script is sourced by another script
# DIRECT_SSH=0

if [ $DIRECT_SSH -ne 0 ]; then
  echo "Using rsync -e ssh"
  SSH_FLAG="-e ssh"
  JUSUF_CODE="judac:data_proc_code"
  JUSUF_BASE="judac:ju_oscbagdis"
  #SLEEP="sleep 1s"
  SLEEP="wait"
  echo "not implemented; need to change _rsync_careful"; exit 1
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
  JUSUF_CODE="$mountpath/data_proc_code"
  JUSUF_BASE="$mountpath"
  SLEEP="wait"
fi

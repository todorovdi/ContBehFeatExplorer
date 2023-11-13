#!/bin/bash
#if [[ $# -lt 1 ]]; then
#  echo "Need at least one arg"
#  exit 1
#fi

DIRECT_SSH=0
. remount_HPC.sh

#OUTPUT_OSCBAGDIS_JUSUF=/p/project/icei-hbp-2020-0012/OSCBAGDIS/output
OUTPUT_OSCBAGDIS_JUSUF_MNT=$JUSUF_BASE/output
t=$HOME/ownCloud/Current
OUTPUT_OWNCLOUD=$t/output/oscbagdis/jusuf
rsync -avz --progress $OUTPUT_OSCBAGDIS_JUSUF_MNT/* $OUTPUT_OWNCLOUD

SUBDIRS=("SSS/" "")
nsubdirs=${#SUBDIRS[*]}

COPY_ANNS=0
COPY_RAW_MAT=0
COPY_SRCD=0
COPY_REC_INFO=0
COPY_LFP=1
DRY=""
#DRY="n"

ALL_DATASETS_STR='S{01,02,03,04,05,06,07,08,09,10}_{on,off}_{hold,move,rest}'

#a = archive - means it preserves permissions (owners, groups), times, symbolic links, and devices.
#r = recursive - means it copies directories and sub directories
#v = verbose - means that it prints on the screen what is being copied
ERRS=--ignore-errors
ERRS=--ignore-missing-args
ERRS=""

OUT_DIR=data_oscbagdis/data_duss

# NON-RECURSIVE copy
if [ $COPY_ANNS -ne 0 ]; then
  rsync -avzh$DRY --progress $ERRS -e ssh $DATA_DUSS/*.txt judac:$OUT_DIR
fi
if [ $COPY_RAW_MAT -ne 0 ]; then
  FNS=`ls -L $DATA_DUSS/S{01,02,03,04,05,06,07,08,09,10}_{on,off}_{hold,move,rest}.mat 2>&1| grep -v "No such"`
  echo $FNS
  nmats=${#FNS[*]}
  echo "Copying Matlab raws"
  #rsync -avzh$DRY --progress $ERRS -e ssh $DATA_DUSS/$ALL_DATASETS_STR.mat judac:data_oscbagdis/
  rsync -avzh$DRY --progress --ignore-existing $ERRS -e ssh $FNS judac:$OUT_DIR
fi

if [ $COPY_LFP -ne 0 ]; then
  echo "Copying LFP fif"
  FNS=`ls -L $DATA_DUSS/*LFP*.fif`
  rsync -avzh$DRY --progress $ERRS -e ssh $FNS judac:$OUT_DIR
  echo "Copying EMG fif"
  FNS=`ls -L $DATA_DUSS/*emg*.fif`
  rsync -avzh$DRY --progress $ERRS -e ssh $FNS judac:$OUT_DIR
fi

for (( i=0; i<=$(( $nsubdirs -1 )); i++ )); do
  SUBDIR=${SUBDIRS[$i]}
  if [ $COPY_REC_INFO -ne 0 ]; then
    echo "rec_info Copying subdir $SUBDIR"
    rsync -avzh$DRY --progress $ERRS -e ssh $DATA_DUSS/$SUBDIR*aal_grp10_src_rec_info* judac:$OUT_DIR/$SUBDIR
  fi
done

for (( i=0; i<=$(( $nsubdirs -1 )); i++ )); do
  SUBDIR=${SUBDIRS[$i]}
  if [ $COPY_SRCD -ne 0 ]; then
    echo "srcd copying subdir $SUBDIR"
    rsync -avzh$DRY --progress $ERRS -e ssh $DATA_DUSS/"$SUBDIR"srcd_$ALL_DATASETS_STR_*parcel_aal.mat judac:$OUT_DIR/$SUBDIR 
  fi
done

#!/bin/bash
#      raws1=(S01_off_hold S01_on_hold S02_off_hold S02_on_hold S04_off_hold S04_on_hold S05_off_hold S05_on_hold S07_off_hold S07_on_hold S03_off_hold)
#raws_compl1=(S01_off_move S01_on_move S02_off_move S02_on_move S04_off_move S04_on_move S05_off_move S05_on_move S07_off_move S07_on_move S03_off_move)
#
## remaining files
#      raws2=(S06_off_hold S06_on_hold S08_off_rest S08_on_rest S09_off_rest S10_off_move S10_off_rest)
#raws_compl2=(S06_off_move S06_on_move)
#
#raws=(${raws1[@]} ${raws2[@]})
#raws_compl=(${raws_compl1[@]} ${raws_compl2[@]})

#raws=(${raws_off[@]} ${raws_on[@]})
#raws_compl=(${raws_compl1[@]})

. $OSCBAGDIS_DATAPROC_CODE/run/def_raw_collections.sh

# order is not important in this case
raws_all=(${raws_off[@]} ${raws_on[@]} ${raws_off_compl[@]} ${raws_on_compl[@]})
raws_all=(${raws_on[@]} ${raws_off_compl[@]} ${raws_on_compl[@]} ${raws_off[@]})
raws_all=(${raws_off_compl[@]})
raws_all=(S02_off_move S03_off_move S04_off_move S05_off_move S07_off_move)

# TMP
#raws_all=(${raws_off0[@]} ${raws_on0[@]} ${raws_off_compl0[@]} ${raws_on_compl0[@]})
#raws_all=(S01_off_move)

#raws_all=(S94_off_hold)

RESAMPLE=1

#######################

#raws_compl=(99_off_move)
#raws=()
#
#raws_compl=(S01_off_hold)
#raws=()

#      raws=(S02_on_hold S04_off_hold S04_on_hold S05_off_hold S05_on_hold  S07_off_hold S07_on_hold S03_off_hold)
#raws_compl=(S02_on_move S04_off_move S04_on_move S05_off_move S05_on_move  S07_off_move S07_on_move S03_off_move)
#
#      raws=(S01_off_hold)
#raws_compl=()
#
#      raws=( S03_off_hold )
#raws_compl=( S03_off_move )

#      raws=( S07_off_hold S07_on_hold)
#raws_compl=( S07_off_move S07_on_move)


#raws=(${raws_off[@]} ${raws_on[@]})
#raws_compl=(${raws_off_compl[@]} ${raws_on_compl[@]}) 

# merge all
nraws_all=${#raws_all[*]}
echo "TOTAL RAWS = $nraws_all"
echo "${raws_all[@]}"

if [ $RESAMPLE -ne 0 ]; then
  scrf="srun_resave_resample.sh"
else
  scrf="srun_resave.sh"
fi

for (( i=0; i<$nraws_all; i++ )); do
  . "$OSCBAGDIS_DATAPROC_CODE/run/$scrf" ${raws_all[$i]}
  if [ $EC -ne 0 ]; then
    echo "Exit code $EC, exiting"
    exit $EC
  fi
done

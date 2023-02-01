#!/bin/bash
echo Starting `basename "$0"`

#      raws1=(S01_off_hold S01_on_hold S02_off_hold S02_on_hold S04_off_hold S04_on_hold S05_off_hold S05_on_hold  S07_off_hold S07_on_hold S03_off_hold)
#raws_compl1=(S01_off_move S01_on_move S02_off_move S02_on_move S04_off_move S04_on_move S05_off_move S05_on_move  S07_off_move S07_on_move S03_off_move)

. $OSCBAGDIS_DATAPROC_CODE/run/def_raw_collections.sh

raws_all=(${raws_off[@]} ${raws_on[@]} ${raws_off_compl[@]} ${raws_on_compl[@]})


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

nraws_all=${#raws_all[*]}

if [ $RESAMPLE -ne 0 ]; then
  scrf="srun_resave_resample.sh"
else
  scrf="srun_resave.sh"
fi
echo "scrf=$scrf"
#exit 1

for (( i=0; i<$nraws_all; i++ )); do
#printf "%s\n" "${raws_all[@]}" > _runstrings_resave.txt
  echo "$OSCBAGDIS_DATAPROC_CODE/run/$scrf ${raws_all[$i]}" >> _runstrings_resave.txt 
done

#raws=(${raws_off[@]} ${raws_on[@]})
#raws_compl=(${raws_off_compl[@]} ${raws_on_compl[@]}) 

# merge all
echo "TOTAL RAWS = $nraws_all"
echo "${raws_all[@]}"


for (( i=0; i<$nraws_all; i++ )); do
  r="$OSCBAGDIS_DATAPROC_CODE/run/$scrf"
  echo $r
  . "$r" ${raws_all[$i]}
  echo EC=$EC
  if [ $EC -ne 0 ]; then
    echo "Exit code $EC, exiting"
    exit $EC
  fi
done

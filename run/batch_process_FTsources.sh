#!/bin/bash
interactive=""
#interactive="-i"

###################################

raws_off=(S01_off_hold S02_off_hold S03_off_hold S04_off_hold S05_off_hold S07_off_hold )  
raws_off_compl=(S01_off_move S02_off_move S03_off_move S04_off_move S05_off_move S07_off_move )  
raws_on=(S01_on_hold S02_on_hold S04_on_hold S05_on_hold S07_on_hold )  
raws_on_compl=(S01_on_move S02_on_move S04_on_move S05_on_move S07_on_move )  

#raws_off=(S07_off_hold )  
#raws_off_compl=(S07_off_move )  
#raws_on=(S07_on_hold )  
#raws_on_compl=(S07_on_move )  

raws=(${raws_off[@]} ${raws_on[@]})
raws_compl=(${raws_off_compl[@]} ${raws_on_compl[@]}) 

raws=(S01_off_hold)
raws_compl=(S01_off_move)

raws=(S99_off_move)
raws_compl=("")


      raws=(S01_off_hold)
raws_compl=(S01_off_move)

      raws=(S01_on_hold S02_off_hold S02_on_hold)
raws_compl=(S01_on_move S02_off_move S02_on_move)

      raws=(S01_off_hold S01_on_hold S02_off_hold S02_on_hold S04_off_hold S04_on_hold S05_off_hold S05_on_hold S07_off_hold S07_on_hold)
raws_compl=(S01_off_move S01_on_move S02_off_move S02_on_move S04_off_move S04_on_move S05_off_move S05_on_move S07_off_move S07_on_move)


      raws=(S01_off_hold)
raws_compl=(S02_off_hold)

nraws=${#raws[*]}

#for t in ${raws[@]}; do
#  #ipython3 $interactive processFTsrc.py -- -r $t
#  ipython3 $interactive run_process_FTsources.py -- -r $t
#done

#GROUPINGS=CB_vs_rest,CBmerged_vs_rest,merged,merged_by_side   
#GROUPINGS="motor-related_vs_CB_vs_rest"
GROUPINGS="all_raw"

#ALG_TYPE='PCA+ICA'
ALG_TYPE='all_sources'

MERGE_WITHIN_MEDCOND=0  # merge requires more careful implementation perhaps

for (( i=0; i<=$(( $nraws -1 )); i++ )); do
  t=${raws[$i]}
  t_compl=${raws_compl[$i]}
  if [ $MERGE_WITHIN_MEDCOND -eq 1 ]; then
    ipython3 $interactive run_process_FTsources.py -- -r $t,$t_compl --groupings $GROUPINGS --alg_type $ALG_TYPE
  else
    ipython3 $interactive run_process_FTsources.py -- -r $t --groupings $GROUPINGS --alg_type $ALG_TYPE
    ipython3 $interactive run_process_FTsources.py -- -r $t_compl --groupings $GROUPINGS --alg_type $ALG_TYPE
  fi
done

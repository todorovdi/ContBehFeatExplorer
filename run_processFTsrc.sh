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

#raws=(S01_off_hold)
#raws_compl=(S01_off_move)

nraws=${#raws[*]}

#for t in ${raws[@]}; do
#  #ipython3 $interactive processFTsrc.py -- -r $t
#  ipython3 $interactive run_process_FTsources.py -- -r $t
#done

for (( i=0; i<=$(( $nraws -1 )); i++ )); do
  t=${raws[$i]}
  t_compl=${raws_compl[$i]}
  ipython3 $interactive run_process_FTsources.py -- -r $t,$t_compl
done

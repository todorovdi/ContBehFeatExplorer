#!/bin/bash
interactive=""

raws1=(S01_off_hold S01_off_move S01_on_hold S01_on_move S02_off_hold S02_off_move S02_on_hold S02_on_move)  
raws2=(S03_off_hold S03_off_move S04_off_hold S04_off_move S04_on_hold S04_on_move)  
raws3=(S05_off_hold S05_off_move S05_on_hold S05_on_move S06_off_hold S06_off_move S06_on_hold S06_on_move)  
raws4=(S07_off_hold S07_off_move S07_on_hold S07_on_move S08_on_rest S08_off_rest S09_off_rest S10_off_move S10_off_rest)  
raws=(${raws1[@]} ${raws2[@]} ${raws3[@]} ${raws4[@]})

###################################

raws=(S07_off_hold)               

#raws2=(S04_off_hold S04_off_move S04_on_hold S04_on_move)  
#raws3=(S05_off_hold S05_off_move S05_on_hold S05_on_move S06_off_hold S06_off_move S06_on_hold S06_on_move)  
#raws4=(S07_off_hold S07_off_move S07_on_hold S07_on_move) 
#raws=(${raws2[@]} ${raws3[@]} ${raws4[@]})
raws=(${raws1[@]} ${raws2[@]})


raws=(S05_off_move)               

raws=(S04_off_hold S04_off_move S04_on_hold S04_on_move)  

for t in ${raws[@]}; do
  ipython3 $interactive processFTsrc.py -- -r $t
done



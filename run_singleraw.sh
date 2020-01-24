#!/bin/bash
raws1=()
raws2=()
raws3=()
raws35=()
raws4=()

#raws1=(S01_off_hold S01_off_move S01_on_hold S01_on_move S02_off_hold S02_off_move S02_on_hold S02_on_move)  
#raws2=(S03_off_hold S03_off_move S04_off_hold S04_off_move S04_on_hold S04_on_move)  
#raws3=(S05_off_hold S05_off_move S05_on_hold S05_on_move S06_off_hold S06_off_move S06_on_hold S06_on_move)  
#raws35=(S06_off_hold S06_off_move S06_on_hold S06_on_move)  

#raws35=(S06_on_hold S06_on_move S06_off_move)  
raws4=(S07_off_hold S07_off_move S07_on_hold S07_on_move S08_on_rest S08_off_rest S09_off_rest S10_off_move S10_off_rest)  
#raws4=(S09_off_rest S10_off_move S10_off_rest)  


raws=(${raws1[@]} ${raws2[@]} ${raws3[@]} ${raws35[@]} ${raws4[@]})

raws=(S08_off_rest)
raws=(S10_off_rest)

interactive=""
#interactive="-i"
updates="--update_stats --update_spec"
#updates="--update_stats"
#updates=""

skipPlot=""
skipPlot="--skipPlot"

echo $raws
for t in ${raws[@]}; do
#    python3 $interactive udus_dataproc.py --rawname $t --singleraw $updates
    ipython3 $interactive udus_dataproc.py -- --rawname $t --singleraw $updates $skipPlot
done

#python3 udus_dataproc.py -i $subjind -m $ -t -s
#python3 udus_dataproc.py -i $subjind -m $ -t -s

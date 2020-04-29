meds="off,on"
tasks="hold,move,rest"
subj_str="1,2,3,4,5,6,7,8,9,10"

#subj_str="5"
#meds="off"
#tasks="move"

#subj_str="1,2,5"

update=""
#update="--update_stats"
#update="--update_stats --update_spec"

subj_str="1"
meds="off"
tasks="hold"

otherside="--plot_other_side"
otherside=""

interactive=""
interactive="-i"

meds="on"
plot_prename="--plot_prename ($meds)_"
plot_prename=""

ipython3 $interactive udus_dataproc.py -- -i $subj_str -m $meds -t $tasks --meanspec $update $otherside $plot_prename

#-m pdb -c continue 

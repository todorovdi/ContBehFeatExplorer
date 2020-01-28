subj_str="1,2,3,4"
meds="on,off"
tasks="hold,move,rest"

meds="off"
tasks="hold"

meds="off,on"
tasks="hold,move,rest"
subj_str="1,2,3,4,5,6,7,8,9,10"
#subj_str="8"
subj_str="1"

tasks="hold"

update=""
#update="--update_stats"

otherside="--plot_other_side"

interactive=""
interactive="-i"
ipython3 $interactive udus_dataproc.py -- -i $subj_str -m $meds -t $tasks --barplot $update $otherside

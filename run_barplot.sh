meds="off,on"
tasks="hold,move,rest"
subj_str="1,2,3,4,5,6,7,8,9,10"

#subj_str="5"
#meds="off"
#tasks="move"

update=""
#update="--update_stats"

#otherside="--plot_other_side"
otherside=""

interactive=""
#interactive="-i"
ipython3 $interactive udus_dataproc.py -- -i $subj_str -m $meds -t $tasks --barplot $update $otherside

#-m pdb -c continue 

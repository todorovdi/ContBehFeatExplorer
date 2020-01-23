subj_str="1,2,3,4"
meds="on,off"
tasks="hold,move,rest"

meds="off"
tasks="hold"
subj_str="1,2,3,4"

interactive=""
interactive="-i"
ipython3 $interactive udus_dataproc.py -- -i $subj_str -m $meds -t $tasks --barplot 

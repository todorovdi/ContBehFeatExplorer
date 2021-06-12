subj=6
dur=30
#dur=100
task="move"
med="off"
interactive=""
interactive="-i"
ipython3 $interactive udus_dataproc.py -- -i $subj -m $med -t $task --singleraw --update_stats --update_spec --spec_time_end=$dur --plot_time_end=$dur --time_end_forstats=$dur

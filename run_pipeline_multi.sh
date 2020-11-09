do_genfeats=$1
do_PCA=$2
do_tSNE=$3
 
allow_multiproc=0


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

#raws=(S04_off_hold)
#raws_compl=(S04_off_move)
#
#raws=(S04_on_hold)
#raws_compl=(S04_on_move)

#raws=(S01_off_hold S01_on_hold)
#raws_compl=(S01_off_move S01_on_move)

#raws=(S01_off_hold)
#raws_compl=(S01_off_move)

nraws=${#raws[*]}

# if we run only PCA/LDA, then we can do it in parallel (because they are not parallalized within)
if [[ $do_genfeats -eq 0 && $do_tSNE -eq 0 && $do_PCA -eq 1 && $allow_multiproc -eq 1 ]]; then
  num_procs=10
  num_jobs="\j"
  for (( i=0; i<=$(( $nraws -1 )); i++ )); do
  #for t in ${raws[@]}; do
    while (( ${num_jobs@P} >= num_procs )); do
      wait -n
    done
    . run_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raw ${raws[$i]} --raw_compl ${raws_compl[$i]}  &
  done
else
  for (( i=0; i<=$(( $nraws -1 )); i++ )); do
    . run_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raw ${raws[$i]} --raw_compl ${raws_compl[$i]}  
  done
fi

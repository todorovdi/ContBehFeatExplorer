if [[ $# -lt 5 ]]; then
  echo "Please put <do_genfeats> <do_PCA> <do_tSNE> <multi_raw_mode> <runstrings_mode>"
  exit 1
fi

do_genfeats=$1
do_PCA=$2
do_tSNE=$3
MULTI_RAW_MODE=$4  # run with arbitrary number of rawnames
SAVE_RUNSTR_MODE=$5
 
allow_multiproc=0  # only makes sense if I want to run single proc one of the bashes



raws_off=(      S01_off_hold S02_off_hold S03_off_hold S04_off_hold S05_off_hold S07_off_hold )  
raws_off_compl=(S01_off_move S02_off_move S03_off_move S04_off_move S05_off_move S07_off_move )  
raws_on=(      S01_on_hold S02_on_hold S04_on_hold S05_on_hold S07_on_hold )  
raws_on_compl=(S01_on_move S02_on_move S04_on_move S05_on_move S07_on_move )  


#raws_off=(      S04_off_hold S05_off_hold S07_off_hold )  
#raws_off_compl=(S04_off_move S05_off_move S07_off_move )  
#raws_on=(      S02_on_hold S04_on_hold S05_on_hold S07_on_hold )  
#raws_on_compl=(S02_on_move S04_on_move S05_on_move S07_on_move )  

#raws_off=(S07_off_hold )  
#raws_off_compl=(S07_off_move )  
#raws_on=(S07_on_hold )  
#raws_on_compl=(S07_on_move )  


raws=(${raws_off[@]} ${raws_on[@]})
raws_compl=(${raws_off_compl[@]} ${raws_on_compl[@]}) 

      raws_decent=(S01_off_hold S01_on_hold S02_off_hold S02_on_hold S04_off_hold S04_on_hold S05_off_hold S05_on_hold S07_off_hold S07_on_hold S03_off_hold)
raws_compl_decent=(S01_off_move S01_on_move S02_off_move S02_on_move S04_off_move S04_on_move S05_off_move S05_on_move S07_off_move S07_on_move S03_off_move)

#      raws=(S01_off_hold S01_on_hold S02_off_hold S02_on_hold S04_off_hold S04_on_hold S07_off_hold S07_on_hold)
#raws_compl=(S01_off_move S01_on_move S02_off_move S02_on_move S04_off_move S04_on_move S07_off_move S07_on_move)
#
#      raws=(S01_off_hold S01_on_hold S02_off_hold S02_on_hold S04_off_hold)
#raws_compl=(S01_off_move S01_on_move S02_off_move S02_on_move S04_off_move)
#
#      raws=(S04_on_hold S07_off_hold S07_on_hold)
#raws_compl=(S04_on_move S07_off_move S07_on_move)

# group all per subject
#      raws=(S01_off_hold S01_on_hold)
#raws_compl=(S01_off_move S01_on_move)
#      raws=(S02_off_hold S02_on_hold)
#raws_compl=(S02_off_move S02_on_move)
#      raws=(S04_off_hold S04_on_hold)
#raws_compl=(S04_off_move S04_on_move)
#      raws=(S07_off_hold S07_on_hold)
#raws_compl=(S07_off_move S07_on_move)

raws=$raws_decent
raws_compl=$raws_compl_decent


raws_strs=("S01_off_hold,S01_on_hold,S01_off_move,S01_on_move" "S02_off_hold,S02_on_hold,S02_off_move,S02_on_move" "S04_off_hold,S04_on_hold,S04_off_move,S04_on_move"  "S05_off_hold,S05_on_hold,S05_off_move,S05_on_move" "S07_off_hold,S07_on_hold,S07_off_move,S07_on_move" "S03_off_hold,S03_off_move")

# no first two
#raws_strs=("S04_off_hold,S04_on_hold,S04_off_move,S04_on_move"  "S05_off_hold,S05_on_hold,S05_off_move,S05_on_move" "S07_off_hold,S07_on_hold,S07_off_move,S07_on_move" "S03_off_hold,S03_off_move")
#raws_strs=("S04_off_hold,S04_on_hold,S04_off_move,S04_on_move"  "S05_off_hold,S05_on_hold,S05_off_move,S05_on_move" "S07_off_hold,S07_on_hold,S07_off_move,S07_on_move" "S03_off_hold,S03_off_move")

#raws_strs=("S07_off_hold,S07_on_hold,S07_off_move,S07_on_move")

#raws_strs=("S02_off_hold,S02_on_hold,S02_off_move,S02_on_move" "S04_off_hold,S04_on_hold,S04_off_move,S04_on_move"  "S07_off_hold,S07_on_hold,S07_off_move,S07_on_move")
#raws_strs=("S05_off_hold,S05_on_hold,S05_off_move,S05_on_move")
#raws_strs=("S99_off_hold,S99_on_hold,S99_off_move,S99_on_move")
#raws_strs=("S03_off_hold,S03_off_move")
#raws_strs=("S01_off_hold,S01_on_hold,S01_off_move")

# everything together to be joined
#raws_strs=("S01_off_hold,S01_on_hold,S01_off_move,S01_on_move,S02_off_hold,S02_on_hold,S02_off_move,S02_on_move,S04_off_hold,S04_on_hold,S04_off_move,S04_on_move,S07_off_hold,S07_on_hold,S07_off_move,S07_on_move" )

MULTI_RAW_STRS_MODE=1
COMPL_RUN_JOINT=0

# differet main sides
#raws=(S01_off_hold)
#raws_compl=(S05_off_hold)

#raws=(S04_off_hold)
#raws_compl=(S04_off_move)
#
#raws=(S04_on_hold)
#raws_compl=(S04_on_move)

#raws=(      S01_off_hold S01_on_hold)
#raws_compl=(S01_off_move S01_on_move)
#
#raws=(      S01_off_hold S01_on_hold S03_off_hold)
#raws_compl=(S01_off_move S01_on_move S03_off_move)
#
#raws=(      S01_off_hold)
#raws_compl=(S01_off_move)
#
#raws=(      S01_on_hold S03_off_hold)
#raws_compl=(S01_on_move S03_off_move)

#raws=(S01_on_hold)
#raws_compl=(S01_on_move)


raws_all=(${raws[@]} ${raws_compl[@]})

nraws=${#raws[*]}
nraws_strs=${#raws_strs[*]}

# create a long string will raws seprarated by commas
raws_str=""
for (( i=0; i<=$(( $nraws -1 )); i++ )); do
  raws_str="${raws[$i]},$raws_str"
  raws_str="${raws_compl[$i]},$raws_str"
done
len=${#raws_str}
len_sub=$(($len-1))
#echo $len
#echo $len_sub
raws_str=${raws_str:0:len_sub}  # get rid of last comma (otherwise I get rawname of zero len and my code will complain)

echo raws_str=$raws_str

RUNSTRING_P_STR=""
RUNSTRINGS_FN="_runstrings.txt"
if [ $SAVE_RUNSTR_MODE -ne 0 ]; then
  echo "SAVE_RUNSTR_MODE=$SAVE_RUNSTR_MODE"
  # empty file (or create new empty)
  >$RUNSTRINGS_FN
  RUNSTRING_P_STR="--runstrings_fn $RUNSTRINGS_FN"
fi
#exit 0

if [ $MULTI_RAW_STRS_MODE -eq 1 && $COMPL_RUN_JOINT -eq 1 ];then
  echo "cannot have both = 1, MULTI_RAW_STRS_MODE and COMPL_RUN_JOINT!!"
fi

#TODO: merge raws and raws_comple and run with COMPL_RUN_JOING = 0

# dot before the command means that file contents get sourced into the shell

# if we run only PCA/LDA, then we can do it in parallel (because they are not parallalized within)
if [[ $do_genfeats -eq 0 && $do_tSNE -eq 0 && $do_PCA -eq 1 && $allow_multiproc -eq 1 ]]; then
  num_procs=`grep -c ^processor /proc/cpuinfo`
  num_jobs="\j"
  for (( i=0; i<=$(( $nraws -1 )); i++ )); do
  #for t in ${raws[@]}; do
    while (( ${num_jobs@P} >= num_procs )); do
      wait -n
    done
    if [ $COMPL_RUN_JOINT -ne 0 ]; then
      . srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raw ${raws[$i]} --raw_compl ${raws_compl[$i]}  &
    else
      . srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raw ${raws[$i]}         &
      . srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raw ${raws_compl[$i]}   &
    fi
  done
else
  if [ $MULTI_RAW_MODE -eq 1 ]; then
    if [ $MULTI_RAW_STRS_MODE -eq 1 ]; then
      # improtant to have disting index name
      for (( rawstri=0; rawstri<$nraws_strs; rawstri++ )); do
        . srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raws_multi ${raws_strs[$rawstri]} $RUNSTRING_P_STR
      done
    else #$MULTI_RAW_STRS_MODE -eq 0
      . srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raws_multi $raws_str $RUNSTRING_P_STR
    fi
  else # $MULTI_RAW_MODE -eq 0
    if [ $COMPL_RUN_JOINT -ne 0 ]; then
      for (( rawi=0; rawi<=$(( $nraws -1 )); rawi++ )); do
        #. srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raw ${raws[$i]} --raw_compl ${raws_compl[$i]}  
          . srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raw ${raws[$rawi]} --raw_compl ${raws_compl[$rawi]} $RUNSTRING_P_STR
      done
    else
      nraws_all=${#raws_all[*]}
      for (( rawi=0; rawi<$nraws_all; rawi++ )); do
        . srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raw ${raws_all[$rawi]}  $RUNSTRING_P_STR        
      done
    fi
  fi
fi

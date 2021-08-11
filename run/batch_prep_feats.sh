RAW_SUBDIR=""
raws=""
SCALE_DATA_COMBINE_TYPE=medcond

raws_strs=("S01_off_hold,S01_on_hold,S01_off_move,S01_on_move" "S02_off_hold,S02_on_hold,S02_off_move,S02_on_move" "S04_off_hold,S04_on_hold,S04_off_move,S04_on_move"  "S05_off_hold,S05_on_hold,S05_off_move,S05_on_move" "S07_off_hold,S07_on_hold,S07_off_move,S07_on_move" "S03_off_hold,S03_off_move")
nraws_strs=${#raws_strs[*]}

#################

if [ ${#RAW_SUBDIR} -gt 0 ]; then
  INPUT_SUBDIR_STR="--input_subdir $RAW_SUBDIR" 
else
  INPUT_SUBDIR_STR=""
fi

export PYTHONPATH=$OSCBAGDIS_DATAPROC_CODE:$PYTHONPATH
MULTI_RAW_STRS_MODE=0
echo MULTI_RAW_STRS_MODE=$MULTI_RAW_STRS_MODE
if [ $MULTI_RAW_STRS_MODE -ne 0 ]; then
  echo "Try to gather stats separately"
  # improtant to have distinct index name
  for (( rawstri=0; rawstri<$nraws_strs; rawstri++ )); do
    #. srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raws_multi ${raws_strs[$rawstri]} $RUNSTRING_P_STR
    # needs all files (LFP,LFPonly,pcica,annotations) in the directory already
    python run_prep_dat.py --param_file prep_dat_defparams.ini $INPUT_SUBDIR_STR --rawnames ${raws_strs[$rawstri]}
  done
else
  echo "Try to gather stats for everyone"
  # needs all files (LFP,LFPonly,pcica,annotations) in the directory already
  # it is dangerous because can take too much mem but it allowes to not care about propagation of stats filenames
  python run_prep_dat.py --param_file prep_dat_defparams.ini $INPUT_SUBDIR_STR  
fi


## make plots
#GENFEATS_PLOT_STR="--show_plots 1 --plot_types raw_stats_scatter,feat_stats_scatter"
#ipython run_genfeats.py -- -r "$raws" --param_file genfeats_defparams.ini $GENFEATS_PLOT_STR $INPUT_SUBDIR_STR --scale_data_combine_type $SCALE_DATA_COMBINE_TYPE --load_only 1
#
# generate for all independently
#./batch_pipeline_HPC.sh 1 0 0 0 1

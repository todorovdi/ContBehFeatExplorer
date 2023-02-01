# if it is given in .ini, and it is nonozero here, there will be error
#RAW_SUBDIR="" 
RAW_SUBDIR="feats_wholectx_LFP256" 

#SRCREC_SUBDIR="SSS_covmat_rest" 
#OUTPUT_SUBDIR="feats_wholectx_LFP256_SSS_covmat_rest"

SRCREC_SUBDIR="SSS_covmat_entire" 
OUTPUT_SUBDIR="feats_wholectx_LFP256_SSS_covmat_entire"

#SRCREC_SUBDIR="hipass_covmat_entire" 
#OUTPUT_SUBDIR="feats_wholectx_LFP256_covmat_entire"

# prescale happens in genfeats, not here, I compute stats for all combine types here anyway
#
#SRCREC_SUBDIR="hipass_covmat_rest" 
#OUTPUT_SUBDIR="feats_wholectx_LFP256_covmat_rest"
#raws=""
SCALE_DATA_COMBINE_TYPE=medcond
#PARAM_FILE=prep_dat_defparams.ini
PARAM_FILE=prep_dat_wholectx_HPC.ini
MULTI_RAW_STRS_MODE=0   # 0 means we will run for everyone

raws_strs=("S01_off_hold,S01_on_hold,S01_off_move,S01_on_move" "S02_off_hold,S02_on_hold,S02_off_move,S02_on_move" "S04_off_hold,S04_on_hold,S04_off_move,S04_on_move"  "S05_off_hold,S05_on_hold,S05_off_move,S05_on_move" "S07_off_hold,S07_on_hold,S07_off_move,S07_on_move" "S03_off_hold,S03_off_move")
nraws_strs=${#raws_strs[*]}

#################

if [ ${#RAW_SUBDIR} -gt 0 ]; then
  INPUT_SUBDIR_STR="--input_subdir $RAW_SUBDIR" 
else
  INPUT_SUBDIR_STR=""
fi

if [ ${#SRCREC_SUBDIR} -gt 0 ]; then
  INPUT_SUBDIR_STR="$INPUT_SUBDIR_STR --input_subdir_srcrec $SRCREC_SUBDIR" 
  echo INPUT_SUBDIR_STR=$INPUT_SUBDIR_STR
fi

if [ ${#OUTPUT_SUBDIR} -gt 0 ]; then
  OUTPUT_SUBDIR_STR="--output_subdir $OUTPUT_SUBDIR" 
else
  OUTPUT_SUBDIR_STR=""
fi

if [ $# -ge 1 ]; then
  NSTOP=$1
else
  NSTOP=$nraws_strs
fi

export PYTHONPATH=$OSCBAGDIS_DATAPROC_CODE:$PYTHONPATH
echo MULTI_RAW_STRS_MODE=$MULTI_RAW_STRS_MODE
if [ $MULTI_RAW_STRS_MODE -ne 0 ]; then
  echo "Try to gather stats separately"
  # improtant to have distinct index name
  for (( rawstri=0; rawstri<NSTOP; rawstri++ )); do
    #. srun_pipeline.sh $do_genfeats $do_PCA $do_tSNE --raws_multi ${raws_strs[$rawstri]} $RUNSTRING_P_STR
    # needs all files (LFP,LFPonly,pcica,annotations) in the directory already
    python $CODE/run/run_prep_dat.py --param_file $PARAM_FILE $INPUT_SUBDIR_STR $OUTPUT_SUBDIR_STR --rawnames ${raws_strs[$rawstri]}
  done
else
  echo "Try to gather stats for everyone"
  # needs all files (LFP,LFPonly,pcica,annotations) in the directory already
  # it is dangerous because can take too much mem but it allowes to not care about propagation of stats filenames
  python $CODE/run/run_prep_dat.py --param_file $PARAM_FILE $INPUT_SUBDIR_STR $OUTPUT_SUBDIR_STR 
fi

echo "Finished for SRCREC_SUBDIR=$SRCREC_SUBDIR"

## make plots
#GENFEATS_PLOT_STR="--show_plots 1 --plot_types raw_stats_scatter,feat_stats_scatter"
#ipython run_genfeats.py -- -r "$raws" --param_file genfeats_defparams.ini $GENFEATS_PLOT_STR $INPUT_SUBDIR_STR --scale_data_combine_type $SCALE_DATA_COMBINE_TYPE --load_only 1
#
# generate for all independently
#./batch_pipeline_HPC.sh 1 0 0 0 1

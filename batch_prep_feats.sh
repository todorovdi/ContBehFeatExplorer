RAW_SUBDIR=""
raws=""
SCALE_DATA_COMBINE_TYPE=medcond


#################

if [ ${#RAW_SUBDIR} -gt 0 ]; then
  INPUT_SUBDIR_STR="--input_subdir $RAW_SUBDIR" 
else
  INPUT_SUBDIR_STR=""
fi

# needs all files (LFP,LFPonly,pcica,annotations) in the directory already
python3 run_prep_dat.py --param_file prep_dat_defparams.ini $INPUT_SUBDIR_STR

# make plots
GENFEATS_PLOT_STR="--show_plots 1 --plot_types raw_stats_scatter,feat_stats_scatter"
run_genfeats.py -- -r "$raws" --param_file genfeats_defparams.ini $GENFEATS_PLOT_STR $INPUT_SUBDIR_STR --scale_data_combine_type $SCALE_DATA_COMBINE_TYPE --load_only 1

./batch_pipeline.sh 1 0 0 0 1

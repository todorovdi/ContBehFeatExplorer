#!/bin/bash
# takes one argument -- rawname
# has to be ran from  run subdir

#$raws="S01_off_hold"
#raw="S97_off_move"
#raw="S05_off_hold"
#raw="S05_off_move"
raw=$1

RESAVE_BAD_CHANS_SSS=no
SRC_REC_BAD_CHANS_SSS=no

RESAVE_MAT_FILE=0
       HIGHPASS=0
           tSSS=0
            SSP=0

# INTERACTIVE="-i"
INTERACTIVE=""

   RECONSTRUCT_SOURCES=1
SRC_REC_AFTER_highpass=1
    SRC_REC_AFTER_tSSS=0
     SRC_REC_AFTER_SSP=0
     SRC_REC_AFTER_ICA=0
        RUN_MATLAB_JOB=1

MIN_DURATION_QUIET_ALLOWED=30
#MIN_DURATION_QUIET_ALLOWED=5  # for shorter test datasets

# DEBUG
RECALC_SRC_COORDS=1
MIN_DURATION_QUIET_ALLOWED=5  # for shorter test datasets
SRCREC_COVMAT_REST_ONLY=0

RECALC_SRC_COORDS=0
ROI_TYPES='"'parcel_aal_surf'"'
#ROI_TYPES='"'HirschPt2011,2013'"'
MEG_artif_types='MEGartif,MEGartif_muscle,MEGartif_ICA'

# for sources extraction
GROUPINGS="all_raw"
ALG_TYPE='all_sources'

RUNF=$OSCBAGDIS_DATAPROC_CODE/run/resave.py

if [ $RESAVE_MAT_FILE -ne 0 ]; then
  #DEBUG
  ipython $INTERACTIVE $RUNF -- -r $raw --read_type '""' --to_perform notch
  #ipython $INTERACTIVE $RUNF -- -r $raw --read_type hires-raw --to_perform notch --recalc_artif 1
  EC=$?
  if [ $EC -ne 0 ]; then
    echo 'Exit code $EC, exiting'
    exit $EC
  fi
fi
if [ $HIGHPASS -ne 0 ]; then
  ipython $INTERACTIVE $RUNF -- -r $raw --read_type notch --to_perform highpass  --recalc_artif 0 
  EC=$?
  if [ $EC -ne 0 ]; then
    echo 'Exit code $EC, exiting'
    exit $EC
  fi
fi
if [ $tSSS -ne 0 ]; then
  #ipython $INTERACTIVE $RUNF -- -r $raw --read_type resample,notch --to_perform tSSS,ICA --badchans_SSS $RESAVE_BAD_CHANS_SSS
  ipython $INTERACTIVE $RUNF -- -r $raw --read_type '""' --to_perform tSSS,ICA --badchans_SSS $RESAVE_BAD_CHANS_SSS --output_subdir SSS --recalc_LFPEMG 0 --recalc_artif 0
  EC=$?
  if [ $EC -ne 0 ]; then
    echo 'Exit code $EC, exiting'
    exit $EC
  fi
fi
if [ $SSP -ne 0 ]; then
  ipython $INTERACTIVE $RUNF -- -r $raw --read_type notch --to_perform SSP,ICA --output_subdir SSP --recalc_artif 0
  EC=$?
  if [ $EC -ne 0 ]; then
    echo 'Exit code $EC, exiting'
    exit $EC
  fi
fi
#ipython $INTERACTIVE $RUNF -- -r $raw --read_type resample --to_perform tSSS,ICA

#afterICA will have to be generated by hand

#ipython $INTERACTIVE $RUNF -- -r $raw --read_type "" --to_perform resample,notch,highpass


#S98_modcoord_parcel_aal.mat
#headmodel_grid_S01.mat       
#headmodel_grid_S01_surf.mat


MATLAB_SCRIPT_PARAMS_DEF=""
MATLAB_SCRIPT_PARAMS_DEF=$MATLAB_SCRIPT_PARAMS_DEF'roi = {'$ROI_TYPES'};'
#MATLAB_SCRIPT_PARAMS_DEF=$MATLAB_SCRIPT_PARAMS_DEF'input_rawname_type = ["resample", "notch", "highpass"];'

if [ $RECONSTRUCT_SOURCES -ne 0 ]; then
  if [ $RECALC_SRC_COORDS -ne 0 ]; then
      #better run separately for all subjects at once, since it is fast 
    . srun_prepSourceCoords.sh    
  fi

  if [ $SRCREC_COVMAT_REST_ONLY -ne 0 ]; then
    s="--ann_types $MEG_artif_types,beh_states"
  else
    s="--ann_types $MEG_artif_types"
  fi
  python run_collect_artifacts.py -r $raw --min_dur $MIN_DURATION_QUIET_ALLOWED $s

  # this way for single file only 
  if [ $SRC_REC_AFTER_highpass -ne 0 ]; then
    if [ $RUN_MATLAB_JOB -ne 0 ]; then
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS_DEF
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'input_rawname_type = ["notch", "highpass"];'
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS"use_data_afterICA=0;"
      vardefstr=""
      #vardefstr='rawnames=["'$raw'"]; '"use_data_afterICA=0;"  
      #matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit()"
      . run/srun_Fieldtrip_srcrec.sh $raw
    fi

    #ipython $INTERACTIVE run_process_FTsources.py -- -r $raw --groupings $GROUPINGS --alg_type $ALG_TYPE
  fi
  if [ $SRC_REC_AFTER_tSSS -ne 0 ]; then
    if [ $RUN_MATLAB_JOB -ne 0 ]; then
      INPUT_SUBDIR=SSS
      OUTPUT_SUBDIR=SSS
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS_DEF
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS"use_data_afterICA=0;"
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'input_subdir="'$INPUT_SUBDIR'";'
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'output_subdir="'$OUTPUT_SUBDIR'";'
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'input_rawname_type=["SSS",  "notch", "highpass"];'
      . run/srun_Fieldtrip_srcrec.sh $raw

      #S='input_rawname_type=["SSS",  "notch", "highpass", "resample"]; input_subdir="SSS"; output_subdir="SSS"; '  
      #vardefstr='rawnames=["'$raw'"]; '$S"use_data_afterICA=0;"  
      #echo "vardefstr=$vardefstr"
      #matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit()"
    fi

    ipython $INTERACTIVE run/run_process_FTsources.py -- -r $raw --groupings $GROUPINGS --alg_type $ALG_TYPE --input_subdir SSS --output_subdir SSS
  fi
  if [ $SRC_REC_AFTER_ICA -ne 0 ]; then
    if [ $RUN_MATLAB_JOB -ne 0 ]; then
      #vardefstr='rawnames=["'$raw'"]; '"use_data_afterICA=1;"  
      #matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit()"
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS_DEF
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS"use_data_afterICA=1;"
      MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'input_rawname_type = ["notch", "highpass"];'
      . run/srun_Fieldtrip_srcrec.sh $raw
    fi

    ipython $INTERACTIVE run/run_process_FTsources.py -- -r $raw --groupings $GROUPINGS --alg_type $ALG_TYPE --input_subdir afterICA --output_subdir afterICA
  fi
fi


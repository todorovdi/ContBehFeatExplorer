#!/bin/bash
#designed to run pipeline for precisely two rawnames
if [[ $# -lt 3 ]]; then
  echo "Please put <do_genfeats> <do_PCA> <do_tSNE>"
  exit 1
fi

do_genfeats=$1
do_PCA=$2
do_tSNE=$3
shift
shift 
shift

echo "srun_pipeline called with args $do_genfeats $do_PCA $do_tSNE"

LOAD_TSNE=0  #allows to skip those that were already computed
skip_XGB=0
#SUBSKIP_ML_FIT=4
SUBSKIP_ML_FIT=2


raw=""
raw_compl=""
# parse arguments
POSITIONAL=()
MULTI_RAW_MODE=0
while [[ $# -gt 0 ]]
do
  key="$1"

  case $key in
      --raw)
      raw="$2"
      shift # past argument
      shift # past value
      ;;
      --raw_compl)
      raw_compl="$2"
      shift # past argument
      shift # past value
      ;;
      --raws_multi)
      raws_multi="$2"
      MULTI_RAW_MODE=1
      shift # past argument
      shift # past value
      ;;
      --LOAD_TSNE)
      LOAD_TSNE="$2"
      shift # past argument
      shift # past value
      ;;
      --SUBSKIP)
      SUBSKIP="$2"
      shift # past argument
      shift # past value
      ;;
      *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done


SINGLE_RAW_MODE=0
if [ $MULTI_RAW_MODE -eq 0 ]; then
  len_raw=${#raw}
  len_raw_compl=${#raw_compl}
  #echo "lens=$len_raw,$len_raw_compl"
  if [[ $len_raw -le 4 && $len_raw_compl -le 4 ]]; then
    echo "no raws provided $raw  $raw_compl"
    exit 1
  fi
  
  if [[ $len_raw -ge 3 && $len_raw_compl -eq 0 ]]; then
    SINGLE_RAW_MODE=1
    echo "single raw=$raw"
  else
    echo "raw=$raw,raw_compl=$raw_compl"
  fi
else
  echo "raws_multi=$raws_multi"
  defIFS=$IFS
  IFS=','
  read -a raws_arr <<< $raws_multi
  IFS=$defIFS
fi

#exit 0

# feats pca tSNE
#do_genfeats=$1
#do_PCA=$2
#do_tSNE=$3
##################

#raw=S01_off_hold
#raw_compl=S01_off_move
#
#raw=S01_on_hold
#raw_compl=S01_on_move
#
#raw=S02_off_hold
#raw_compl=S02_off_move
#
#raw=S02_on_hold
#raw_compl=S02_on_move
#
#raw=S03_off_hold
#raw_compl=S03_off_move

interactive=""
#interactive="-i"

single_core=0
KALMAN=0
PCA_LOAD_ONLY=0 # useful I want to just plot feat stats

#SOURCES_TYPE=""
SOURCES_TYPE="parcel_aal"

DESIRED_PCA_EXPLAIN=0.95
DESIRED_DISCARD=0.01
PCA_PLOTS=1
PCA_PLOT_TYPES=feat_stats

HEAVY_FIT_REDUCED_FEATSET=0


# tSNE run params
DO_TSNE_USING_INDIVID_PCA=0
DO_TSNE_USING_COMMON_PCA=1
#dim_inp_tSNE=60
#dim_inp_tSNE=80
#dim_inp_tSNE=100
dim_inp_tSNE=-1
SUBSKIP=1
TSNE_PLOTS=1
#raws=$raw,$raw_compl

if [ $SINGLE_RAW_MODE -eq 1 ]; then
  DO_RAW=1
  DO_RAW_COMPL=0
  DO_RAW_BOTH=0

  #DO_RAW_PCA=1
  #DO_RAW_COMPL_PCA=0
  #DO_RAW_BOTH_PCA=0
else
  DO_RAW=0
  DO_RAW_COMPL=0
  DO_RAW_BOTH=1

  #DO_RAW_PCA=0
  #DO_RAW_COMPL_PCA=0
  #DO_RAW_BOTH_PCA=1
fi

PREFIX_ALL_FEATS=0
   PREFIXES_MAIN=1
PREFIXES_AUX_SRC=0
    PREFIXES_AUX=0
LFP_CHAN_TO_USE="all"
#LFP_CHAN_TO_USE="main"
#BANDS_TYPE="fine"
BANDS_TYPE="crude"
FEAT_TYPES_TO_USE="con,H_act,H_mob,H_compl,rbcorr,bpcorr"    # for run_genfeats only
SRC_GROUPING=0      # index of grouping inside the file
#SRC_GROUPING_FN=9
SRC_GROUPING_FN=10

USE_AUX_GROUPINGS=1

RAW_SUBDIR=""
FEAT_SUBDIR=""

#--groupings_to_use
#GROUPINGS_TO_USE="merge_all_not_trem,merge_movements,merge_nothing"
GROUPINGS_TO_USE="merge_nothing,merge_all_not_trem"

#int_types_to_use = gp.int_types_to_include
INT_TYPES_TO_USE="basic,trem_vs_quiet"

if [ $USE_AUX_GROUPINGS -gt 0 ]; then
  GROUPINGS_TO_USE="$GROUPINGS_TO_USE,merge_within_subj,merge_within_medcond,merge_within_task"
  INT_TYPES_TO_USE="$INT_TYPES_TO_USE,subj_medcond_task,subj_medcond,subj"
fi


# for S01_off_hold
#CROP="170,220"
CROP=","

if [ $MULTI_RAW_MODE -eq 0 ]; then
  if [ $SINGLE_RAW_MODE -eq 1 ]; then 
    raws="$raw"
  else
    raws="$raw,$raw_compl"
  fi
else
  raws="$raws_multi"
fi
#raws="$raw"

# for genfeats
SAVE_TFR=1
SAVE_CSD=1
LOAD_TFR=0
LOAD_CSD=0
SAVE_RBCORR=1
SAVE_BPCORR=1

MAX_TFR_AGE_H=240 # 10 days, it was gen on Jan 15
 
PARCEL_TYPES_CB="Cerebellum"
PARCEL_TYPES_NONMOTOR="not_motor-related"
PARCEL_TYPES_MOTOR="motor-related" 

if [[ $BANDS_TYPE == "fine" ]]; then
  BANDS_BETA=low_beta,high_beta                  
  BANDS_GAMMA=low_gamma,high_gamma                   
  BANDS_HFO=HFO1,HFO2,HFO3                   
else
  BANDS_BETA=beta                  
  BANDS_GAMMA=gamma                   
  BANDS_HFO=HFO                   
fi

if [ $do_genfeats -gt 0 ]; then
  GENFEATS_INTERMED="--load_TFR $LOAD_TFR --load_CSD $LOAD_CSD --save_TFR $SAVE_TFR --save_CSD $SAVE_CSD --save_rbcorr $SAVE_RBCORR --save_bpcorr $SAVE_BPCORR --load_TFRCSD_max_age_h $MAX_TFR_AGE_H"
  if [ ${#RAW_SUBDIR} -gt 0 ]; then
    INPUT_SUBDIR_STR="--input_subdir $RAW_SUBDIR" 
  else
    INPUT_SUBDIR_STR=""
  fi
  if [ ${#FEAT_SUBDIR} -gt 0 ]; then
    OUTPUT_SUBDIR_STR="--output_subdir $FEAT_SUBDIR"
  else
    OUTPUT_SUBDIR_STR=""
  fi
  SUBDIR_STR="$INPUT_SUBDIR_STR $OUTPUT_SUBDIR_STR"
  GENFEATS_PLOT_STR="--show_plots 1 --plot_types raw_stats_scatter,feat_stats_scatter"
  ipython3 $interactive run_genfeats.py -- -r "$raws" --bands "$BANDS_TYPE" --sources_type "$SOURCES_TYPE" --feat_types "$FEAT_TYPES_TO_USE" --crop "$CROP" --src_grouping "$SRC_GROUPING" --src_grouping_fn $SRC_GROUPING_FN --Kalman_smooth $KALMAN $GENFEATS_INTERMED $GENFEATS_PLOT_STR
fi

if [ $do_PCA -gt 0 ]; then

  #ipython3 $interactive run_PCA.py  -- -r $raw,$raw_compl

  function PCA { 
    RS="$interactive run_PCA.py -- $1"
    #echo $RS --prefix all
    #echo RS= $RS
    # use everything (from main trem side)

    if [ $PREFIX_ALL_FEATS -gt 0 ]; then
      ipython3 $RS --search_best_LFP 0           --prefix all 
    fi
    if [ $PREFIXES_MAIN -gt 0 ]; then
      # searching for best LFP with all features takes too much time and makes little sense because 
      # there is too much redundancy
      if [ $? -ne 0 ]; then  #checking exit code
        echo "trying to run with all features resulted in an error, exiting"
        exit 1
      fi
      ipython3 $RS --mods LFP                    --prefix modLFP            
      ipython3 $RS --mods msrc                   --prefix modSrc                             
      ipython3 $RS --fbands $BANDS_BETA          --prefix allb_beta            
      ipython3 $RS --LFP_related_only 1 --cross_couplings_only 1   --prefix LFPrel_noself        
      ipython3 $RS --LFP_related_only 1          --prefix LFPrel                                    
      ipython3 $RS --fbands $BANDS_GAMMA         --prefix allb_gamma           
      ipython3 $RS --fbands tremor               --prefix allb_trem                                  
      ipython3 $RS --fbands $BANDS_HFO           --prefix allb_HFO                                  
    fi
    if [ $PREFIXES_AUX_SRC -gt 0 ]; then
      ipython3 $RS --parcel_group_names $PARCEL_TYPES_MOTOR       --prefix onlyMotorSrc      
      ipython3 $RS --parcel_group_names $PARCEL_TYPES_NONMOTOR    --prefix onlyRestSrc       
      ipython3 $RS --parcel_types $PARCEL_TYPES_CB                --prefix onlyCBSrc         
    fi
    if [ $PREFIXES_AUX -gt 0 ]; then
      ipython3 $RS --feat_types con                        --prefix onlyCon                         
      ipython3 $RS --use_HFO 0                             --prefix allnoHFO                           
      ipython3 $RS --fbands tremor,$BANDS_BETA             --prefix allb_trembeta   
      ipython3 $RS --feat_types H_act,H_mob,H_compl,rbcorr --prefix onlyTD            
      ipython3 $RS --feat_types con,bpcorr                 --prefix onlyFD                          
      ipython3 $RS --feat_types con,H_act,H_mob,H_compl    --prefix conH                 
      ipython3 $RS --feat_types H_act,H_mob,H_compl        --prefix onlyH                    
      ipython3 $RS --feat_types bpcorr                     --prefix onlyBpcorr                      
      ipython3 $RS --feat_types bpcorr  --use_HFO 0        --prefix onlyBpcorrNoHFO     
      ipython3 $RS --feat_types rbcorr                     --prefix onlyRbcorr                      
      ipython3 $RS --feat_types con --use_HFO 0            --prefix onlyConNoHFO        
      ipython3 $RS --mods LFP --use_HFO 0                  --prefix modLFPnoHFO 
    fi
  }


  # for PCA we use same
  if [ ${#FEAT_SUBDIR} -gt 0 ]; then
    INPUT_SUBDIR_STR="--input_subdir $FEAT_SUBDIR"
    OUTPUT_SUBDIR_STR="--output_subdir $FEAT_SUBDIR"
  else
    INPUT_SUBDIR_STR=""
    OUTPUT_SUBDIR_STR=""
  fi
  SUBDIR_STR="$INPUT_SUBDIR_STR $OUTPUT_SUBDIR_STR"
  COMMON_ALL="--pcexpl $DESIRED_PCA_EXPLAIN --discard $DESIRED_DISCARD --show_plots $PCA_PLOTS --sources_type $SOURCES_TYPE --single_core $single_core --bands_type $BANDS_TYPE --src_grouping $SRC_GROUPING --src_grouping_fn $SRC_GROUPING_FN --groupings_to_use=$GROUPINGS_TO_USE --int_types_to_use=$INT_TYPES_TO_USE --skip_XGB $skip_XGB --subskip_fit $SUBSKIP_ML_FIT --LFPchan $LFP_CHAN_TO_USE --heavy_fit_red_featset $HEAVY_FIT_REDUCED_FEATSET $SUBDIR_STR --plot_types $PCA_PLOT_TYPES --load_only $PCA_LOAD_ONLY"

  #echo COMMON_ALL= $COMMON_ALL    # replaces comma with space
  #echo "COMMON_ALL= $COMMON_ALL"
  #echo COMMON_ALL= "$COMMON_ALL"
  if [ $DO_RAW_BOTH -gt 0 ]; then
    COMMON_PART="-r $raws $COMMON_ALL"
    #echo "COMMON_PART= $COMMON_PART"
    #./subrun_pipeline_PCA.sh
    PCA "$COMMON_PART"
  fi

  if [ $DO_RAW -gt 0 ]; then
    COMMON_PART="-r $raw $COMMON_ALL"
    #./subrun_pipeline_PCA.sh
    PCA "$COMMON_PART"
  fi

  if [ $DO_RAW_COMPL -gt 0 ]; then
    COMMON_PART="-r $raw_compl $COMMON_ALL"
    #./subrun_pipeline_PCA.sh
    PCA "$COMMON_PART"
  fi

fi

if [ $do_tSNE -gt 0 ]; then
  #ipython3 $interactive run_tSNE.py --     -r $raw

  function tSNE {
    RS="$interactive run_tSNE.py -- $1"
    if [ $PREFIX_ALL_FEATS -gt 0 ]; then
      ipython3 $RS --prefix all                                                           
    fi
    if [ $PREFIXES_MAIN -gt 0 ]; then
      if [ $? -ne 0 ]; then
        exit 1
      fi
      ipython3 $RS --prefix LFPrel                  
      ipython3 $RS --prefix LFPrel_noself                  
      ipython3 $RS --prefix modLFP                                           
      ipython3 $RS --prefix modSrc                             
      ipython3 $RS --prefix allb_trem                             
      ipython3 $RS --prefix allb_beta       
      ipython3 $RS --prefix allb_gamma      
      ipython3 $RS --prefix allb_HFO      
    fi
    if [ $PREFIXES_AUX_SRC -gt 0 ]; then
      ipython3 $RS --prefix onlyMotorSrc   
      ipython3 $RS --prefix onlyRestSrc    
      ipython3 $RS --prefix onlyCBSrc      
    fi
    if [ $PREFIXES_AUX -gt 0 ]; then
      ipython3 $RS --prefix allnoHFO                 
      ipython3 $RS --prefix allb_trembeta   
      ipython3 $RS --prefix modLFPnoHFO 
      ipython3 $RS --prefix onlyTD            
      ipython3 $RS --prefix onlyFD                          
      ipython3 $RS --prefix conH                 
      ipython3 $RS --prefix onlyH                    
      ipython3 $RS --prefix onlyBpcorr                      
      ipython3 $RS --prefix onlyBpcorrNoHFO     
      ipython3 $RS --prefix onlyRbcorr                      
      ipython3 $RS --prefix onlyCon                         
      ipython3 $RS --prefix onlyConNoHFO        
    fi
    #ipython3 $RS --prefix all             
    #ipython3 $RS --prefix modLFP          
    #if [ $PREFIXES_AUX -gt 0 ]; then
    #  ipython3 $RS --prefix modSrc          
    #  ipython3 $RS --prefix allnoHFO             

    #  ipython3 $RS --prefix allb_trem       
    #  ipython3 $RS --prefix allb_beta       
    #  ipython3 $RS --prefix allb_gamma      
    #  ipython3 $RS --prefix allb_trembeta   

    #  ipython3 $RS --prefix onlyTD          
    #  ipython3 $RS --prefix onlyFD          
    #  ipython3 $RS --prefix onlyH           
    #  ipython3 $RS --prefix conH            
    #  ipython3 $RS --prefix modLFPnoHFO          
    #  ipython3 $RS --prefix onlyRbcorr      
    #  ipython3 $RS --prefix onlyCon         
    #  ipython3 $RS --prefix onlyBpcorrNoHFO 
    #  ipython3 $RS --prefix onlyConNoHFO    
    #  ipython3 $RS --prefix onlyBpcorr      
    #fi
  }


    # we use same
    if [ ${#FEAT_SUBDIR} -gt 0 ]; then
      INPUT_SUBDIR_STR="--input_subdir $FEAT_SUBDIR"
      OUTPUT_SUBDIR_STR="--output_subdir $FEAT_SUBDIR"
    else
      INPUT_SUBDIR_STR=""
      OUTPUT_SUBDIR_STR=""
    fi
    SUBDIR_STR="$INPUT_SUBDIR_STR $OUTPUT_SUBDIR_STR"
    RUNSTRING_COMMON="--subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE $SUBDIR_STR"
  if [ $DO_TSNE_USING_INDIVID_PCA -gt 0 ]; then
    NRAWS_PCA=1
    RUNSTRING="--nrPCA $NRAWS_PCA $RUNSTRING_COMMON"
    if [ $DO_RAW -gt 0 ]; then
      COMMON_PART="-r $raw $RUNSTRING"
      tSNE "$COMMON_PART"
    fi

    if [ $DO_RAW_COMPL -gt 0 ]; then
      COMMON_PART="-r $raw_compl $RUNSTRING"
      tSNE "$COMMON_PART"
    fi

    if [ $DO_RAW_BOTH -gt 0 ]; then
      COMMON_PART="-r $raws $RUNSTRING"
      tSNE "$COMMON_PART"
    fi
  fi

  if [ $DO_TSNE_USING_COMMON_PCA -gt 0 ]; then
    #NRAWS_PCA=2
    NRAWS_PCA=${#raws_arr[*]}
    echo NRAWS_PCA=$NRAWS_PCA
    RUNSTRING="--nrPCA $NRAWS_PCA $RUNSTRING_COMMON"
    if [ $DO_RAW -gt 0 ]; then
      COMMON_PART="-r $raw $RUNSTRING"
      tSNE "$COMMON_PART"
    fi

    if [ $DO_RAW_COMPL -gt 0 ]; then
      COMMON_PART="-r $raw_compl $RUNSTRING"
      tSNE "$COMMON_PART"
    fi

    if [ $DO_RAW_BOTH -gt 0 ]; then
      COMMON_PART="-r $raws $RUNSTRING"
      tSNE "$COMMON_PART"
    fi
  fi

fi




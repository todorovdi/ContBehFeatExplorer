#!/bin/bash


#designed to run pipeline for precisely two rawnames
if [[ $# -lt 3 ]]; then
  echo "Please put <do_genfeats> <do_ML> <do_nlproj>"
  exit 1
fi

do_genfeats=$1
do_ML=$2
do_nlproj=$3
shift
shift 
shift

echo "srun_pipeline called with args $do_genfeats $do_ML $do_nlproj"

SAVE_RUNSTR_MODE=0

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
      --runstrings_fn)
      RUNSTRINGS_FN="$2"
      len_rsfn=${#RUNSTRINGS_FN} 
      if [ $len_rsfn -ge 1 ]; then
        SAVE_RUNSTR_MODE=1
        echo "Using RUNSTRINGS mode with file $RUNSTRINGS_FN"
      fi
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

#raw=S01_off_hold
#raw_compl=S01_off_move

############ common settings
interactive=""
#interactive="-i"

#EXECSTR=ipython3 $interactive
#EXECSTR_FIN=""
#EXECSTR="echo"
#EXECSTR_FIN=" >> _runstrings.txt"

#SOURCES_TYPE=""
SOURCES_TYPE="parcel_aal"
BANDS_TYPE="crude"

#BANDS_TYPE="fine"
SRC_GROUPING=0      # index of grouping inside the file
#SRC_GROUPING_FN=9
SRC_GROUPING_FN=10

#RAW_SUBDIR="SSS"
#FEAT_SUBDIR="SSS"
RAW_SUBDIR=""
#FEAT_SUBDIR="genfeats_scale_combine_subj"
FEAT_SUBDIR=""


MAX_TFR_AGE_H=240 # 10 days, it was gen on Jan 15

if [ $SINGLE_RAW_MODE -eq 1 ]; then
  DO_RAW=1
  DO_RAW_COMPL=0
  DO_RAW_BOTH=0

  #DO_RAW_ML=1
  #DO_RAW_COMPL_ML=0
  #DO_RAW_BOTH_ML=0
else
  DO_RAW=0
  DO_RAW_COMPL=0
  DO_RAW_BOTH=1

  #DO_RAW_ML=0
  #DO_RAW_COMPL_ML=0
  #DO_RAW_BOTH_ML=1
fi

if [ $MULTI_RAW_MODE -eq 0 ]; then
  if [ $SINGLE_RAW_MODE -eq 1 ]; then 
    raws="$raw"
  else
    raws="$raw,$raw_compl"
  fi
else
  raws="$raws_multi"
fi

####################  genfeats params
# for genfeats
LOAD_TFR=0
LOAD_CSD=0

SCALE_DATA_COMBINE_TYPE="medcond"
#SCALE_DATA_COMBINE_TYPE="subj"

################## ML run params
ML_LOAD_ONLY=0 # useful I want to just plot feat stats
DESIRED_PCA_EXPLAIN=0.95
DESIRED_DISCARD=0.01
ML_PLOTS=1
ML_PLOT_TYPES=feat_stats
#
HEAVY_FIT_REDUCED_FEATSET=0
#SUBSKIP_ML_FIT=4
#SUBSKIP_ML_FIT=2
SUBSKIP_ML_FIT=1

LFP_CHAN_TO_USE="all"
#LFP_CHAN_TO_USE="main"
USE_AUX_IVAL_GROUPINGS=1

skip_XGB=0
skip_XGB_aux_int=0
# default val is 30, the smaller the longer the computation takes
max_XGB_step_nfeats=5

if [ $SAVE_RUNSTR_MODE -ne 0 ]; then
  RUN_CLASS_LAB_GRP_SEPARATE=1
else
  RUN_CLASS_LAB_GRP_SEPARATE=0
fi

######## common for ML and nlproj
      PREFIX_ALL_FEATS=1
         PREFIXES_MAIN=1
PREFIXES_CROSS_MOD_AUX=1
      PREFIXES_AUX_SRC=1
          PREFIXES_AUX=0

#--groupings_to_use
#GROUPINGS_TO_USE="merge_all_not_trem,merge_movements,merge_nothing"
GROUPINGS_TO_USE="merge_nothing,merge_all_not_trem"

#int_types_to_use = gp.int_types_to_include
INT_SETS_TO_USE="basic,trem_vs_quiet"

if [ $USE_AUX_IVAL_GROUPINGS -gt 0 ]; then
  GROUPINGS_TO_USE="$GROUPINGS_TO_USE,merge_within_subj,merge_within_medcond,merge_within_task"
  INT_SETS_TO_USE="$INT_SETS_TO_USE,subj_medcond_task,subj_medcond,subj"
fi

echo "GROUPINGS_TO_USE=$GROUPINGS_TO_USE"
echo "INT_SETS_TO_USE=$INT_SETS_TO_USE"
defIFS=$IFS
IFS=','
read -a groupings_arr <<< $GROUPINGS_TO_USE
read -a int_sets_arr <<< $INT_SETS_TO_USE
IFS=$defIFS


PARCEL_TYPES_CB="Cerebellum"
PARCEL_TYPES_NONMOTOR="not_motor-related"
PARCEL_TYPES_MOTOR="motor-related" 

################# nlproj run params
DO_TSNE_USING_INDIVID_ML=0
DO_TSNE_USING_COMMON_ML=1
#dim_inp_nlproj=100
dim_inp_nlproj=-1
SUBSKIP=1
TSNE_PLOTS=1
#raws=$raw,$raw_compl
LOAD_TSNE=0  #allows to skip those that were already computed

#IPYTHON_SYNTAX=""
IPYTHON_SYNTAX="--"
if [ ${#GENFEATS_PARAM_FILE} -eq 0 ]; then
  GENFEATS_PARAM_FILE=genfeats_defparams.ini
fi
if [ ${#MI_PARAM_FILE} -eq 0 ]; then
  MI_PARAM_FILE=MI_defparams.ini
fi
if [ ${#NLPROJ_PARAM_FILE} -eq 0 ]; then
  NLPROJ_PARAM_FILE=nlproj_defparams.ini
fi

echo GENFEATS_PARAM_FILE=$GENFEATS_PARAM_FILE

# FAST TEST ONLY
#GROUPINGS_TO_USE="merge_nothing,merge_all_not_trem"
#INT_SETS_TO_USE="trem_vs_quiet"
#SUBSKIP_ML_FIT=8


# for S01_off_hold
#CROP="170,220"
CROP=","

#raws="$raw"

#exit 1
 

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
  GENFEATS_INTERMED="--load_TFR $LOAD_TFR --load_CSD $LOAD_CSD --load_TFRCSD_max_age_h $MAX_TFR_AGE_H"
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
  #echo SUBDIR_STR=$SUBDIR_STR
  #exit 0
  RUNSTRING_CUR=' run_genfeats.py $IPYTHON_SYNTAX -r "$raws" --param_file $GENFEATS_PARAM_FILE $GENFEATS_INTERMED  $SUBDIR_STR --scale_data_combine_type $SCALE_DATA_COMBINE_TYPE'
  #GENFEATS_PLOT_STR="--show_plots 1 --plot_types raw_stats_scatter,feat_stats_scatter"
  #RUNSTRING_CUR=' run_genfeats.py -- -r "$raws" --param_file $GENFEATS_PARAM_FILE --bands "$BANDS_TYPE" --sources_type "$SOURCES_TYPE" --feat_types "$FEAT_TYPES_TO_USE_GENFEATS" --crop "$CROP" --src_grouping "$SRC_GROUPING" --src_grouping_fn $SRC_GROUPING_FN --Kalman_smooth $KALMAN $GENFEATS_INTERMED $GENFEATS_PLOT_STR --rbcorr_use_local_means $RBCORR_USE_LOCAL_MEANS $SUBDIR_STR --scale_data_combine_type $SCALE_DATA_COMBINE_TYPE'
  R=$(eval echo $RUNSTRING_CUR)
  echo "Current runstring is "$R
  if [ $SAVE_RUNSTR_MODE -eq 0 ]; then
    #R=""
    #eval R=\$$RUNSTRING_CUR
    #R=${!RUNSTRING_CUR}
    #ipython3 $(eval echo \$$RUNSTRING_CUR)
    ipython3 $interactive $R
  else
    echo $R >> $RUNSTRINGS_FN
  fi
fi

if [ $do_ML -gt 0 ]; then

  #$EXECSTR $interactive run_ML.py  -- -r $raw,$raw_compl

  function ML { 
    RS="run_ML.py -- $1"
    #echo $RS --prefix all
    #echo RS= $RS

    # FAST TEST ONLY
    #$EXECSTR $RS --mods LFP                               --prefix modLFP            
    RUNSTRINGS=()

    if [ $PREFIX_ALL_FEATS -gt 0 ]; then
      # use everything (from main trem side)
      RUNSTRING_CUR=' $RS --search_best_LFP 0           --prefix all'
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi
    if [ $PREFIXES_MAIN -gt 0 ]; then
      # searching for best LFP with all features takes too much time and makes little sense because 
      # there is too much redundancy
      if [ $? -ne 0 ]; then  #checking exit code
        echo "trying to run with all features resulted in an error, exiting"
        exit 1
      fi
      RUNSTRING_CUR=' $RS --mods LFP                               --prefix modLFP               '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --self_couplings_only 1 --mods msrc      --prefix modSrc_self          '    
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --mods msrc                              --prefix modSrc               '              
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --fbands $BANDS_BETA                     --prefix allb_beta            '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --LFP_related_only 1 --cross_couplings_only 1   --prefix LFPrel_noself '       
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi
    if [ $PREFIXES_AUX_SRC -gt 0 ]; then
      RUNSTRING_CUR=' $RS --parcel_group_names $PARCEL_TYPES_MOTOR       --prefix onlyMotorSrc  '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --parcel_group_names $PARCEL_TYPES_NONMOTOR    --prefix onlyRestSrc   '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --parcel_types $PARCEL_TYPES_CB                --prefix onlyCBSrc     '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi
    if [ $PREFIXES_CROSS_MOD_AUX -ne 0 ]; then 
      CROSS_MOD="--LFP_related_only 1 --cross_couplings_only 1"
      RUNSTRING_CUR=' $RS $CROSS_MOD --feat_types con             --prefix LFPrel_noself_onlyCon     '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $CROSS_MOD --feat_types rbcorr          --prefix LFPrel_noself_onlyRbcorr  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $CROSS_MOD --feat_types bpcorr          --prefix LFPrel_noself_onlyBpcorr  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi 
    if [ $PREFIXES_AUX -gt 0 ]; then
      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl        --prefix onlyH           '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --LFP_related_only 1                    --prefix LFPrel          '           
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --fbands $BANDS_GAMMA                   --prefix allb_gamma      '    
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --fbands tremor                         --prefix allb_trem       '                      
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --fbands $BANDS_HFO                     --prefix allb_HFO        '                    
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types con                        --prefix onlyCon         '             
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --fbands tremor,$BANDS_BETA             --prefix allb_trembeta   '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl,rbcorr --prefix onlyTD          ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types con,bpcorr                 --prefix onlyFD          '               
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types bpcorr                     --prefix onlyBpcorr      '               
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types bpcorr  --use_HFO 0        --prefix onlyBpcorrNoHFO '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types rbcorr                     --prefix onlyRbcorr      '               
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --mods LFP --use_HFO 0                  --prefix modLFPnoHFO     '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types con,H_act,H_mob,H_compl    --prefix conH            '    
      RUNSTRINGS+=("$RUNSTRING_CUR")
      #$EXECSTR $RS --use_HFO 0                             --prefix allnoHFO                           
      #$EXECSTR $RS --feat_types con --use_HFO 0            --prefix onlyConNoHFO        
    fi

    num_runstr=${#RUNSTRINGS[*]}
    echo "ML: will run in total $num_runstr runstrings"
    for (( i=0; i < $num_runstr; i++ )); do
      RUNSTRING_CUR=${RUNSTRINGS[$i]}
      R=$(eval echo $RUNSTRING_CUR)
      #echo "Current runstring is "$R
      if [ $SAVE_RUNSTR_MODE -eq 0 ]; then
        #R=""
        #eval R=\$$RUNSTRING_CUR
        #R=${!RUNSTRING_CUR}
        #ipython3 $(eval echo \$$RUNSTRING_CUR)
        ipython3 $interactive $R
      else
        echo $R >> $RUNSTRINGS_FN
      fi
    done
  }


  # for ML we use same
  if [ ${#FEAT_SUBDIR} -gt 0 ]; then
    INPUT_SUBDIR_STR="--input_subdir $FEAT_SUBDIR"
    OUTPUT_SUBDIR_STR="--output_subdir $FEAT_SUBDIR"
  else
    INPUT_SUBDIR_STR=""
    OUTPUT_SUBDIR_STR=""
  fi
  SUBDIR_STR="$INPUT_SUBDIR_STR $OUTPUT_SUBDIR_STR"
  #COMMON_ALL="--pcexpl $DESIRED_PCA_EXPLAIN --discard $DESIRED_DISCARD --show_plots $ML_PLOTS --sources_type $SOURCES_TYPE --bands_type $BANDS_TYPE --src_grouping $SRC_GROUPING --src_grouping_fn $SRC_GROUPING_FN --skip_XGB $skip_XGB --skip_XGB_aux_int $skip_XGB_aux_int --max_XGB_step_nfeats $max_XGB_step_nfeats --subskip_fit $SUBSKIP_ML_FIT --LFPchan $LFP_CHAN_TO_USE --heavy_fit_red_featset $HEAVY_FIT_REDUCED_FEATSET $SUBDIR_STR --plot_types $ML_PLOT_TYPES --load_only $ML_LOAD_ONLY --param_file $MI_PARAM_FILE"
  COMMON_ALL="--param_file $MI_PARAM_FILE"
  if [ $RUN_CLASS_LAB_GRP_SEPARATE -eq 0 ]; then
    CLASS_LAB_GRP_STR="--groupings_to_use=$GROUPINGS_TO_USE --int_types_to_use=$INT_SETS_TO_USE"
    #echo COMMON_ALL= $COMMON_ALL    # replaces comma with space
    #echo "COMMON_ALL= $COMMON_ALL"
    #echo COMMON_ALL= "$COMMON_ALL"
    if [ $DO_RAW_BOTH -gt 0 ]; then
      COMMON_PART="-r $raws $COMMON_ALL $CLASS_LAB_GRP_STR"
      #echo "COMMON_PART= $COMMON_PART"
      #./subrun_pipeline_ML.sh
      ML "$COMMON_PART"
    fi

    if [ $DO_RAW -gt 0 ]; then
      COMMON_PART="-r $raw $COMMON_ALL $CLASS_LAB_GRP_STR"
      #./subrun_pipeline_ML.sh
      ML "$COMMON_PART"
    fi

    if [ $DO_RAW_COMPL -gt 0 ]; then
      COMMON_PART="-r $raw_compl $COMMON_ALL $CLASS_LAB_GRP_STR"
      #./subrun_pipeline_ML.sh
      ML "$COMMON_PART"
    fi
  else
    ngrp=${#groupings_arr[*]}
    nints=${#int_sets_arr[*]}
    echo "ngrp=$ngrp,  GROUPINGS=${groupings_arr[*]}"
    for (( ii=0; ii<$ngrp; ii++ )); do
      for (( jj=0; jj<$nints; jj++ )); do
        echo "ngrp=$ngrp,  ii=$ii, curgrp=${groupings_arr[$ii]}"
        GRP=${groupings_arr[$ii]}
        ISET=${int_sets_arr[$jj]}
        CLASS_LAB_GRP_STR="--groupings_to_use $GRP --int_types_to_use $ISET"
        S="import globvars;print(int(\"$GRP\" in globvars.gp.group_vs_int_type_allowed[\"$ISET\"]))"
        echo $S
        #python3 -c "$S" > $OUT
        #python3 -c "$S"
        OUT=$(python3 -c "$S")
        # first line is hostname so take the last line only
        OUT=`echo "$OUT" | tail -n1`
        #echo "OUTTTT $OUT"
        #echo -e "import sys\nfor r in range(10): print 'rob'" | python3
        echo "CLASS_LAB_GRP_STR=$CLASS_LAB_GRP_STR"
        if [[ $DO_RAW_BOTH -gt 0 && $OUT -ne 0 ]]; then
          COMMON_PART="-r $raws $COMMON_ALL $CLASS_LAB_GRP_STR"
          ML "$COMMON_PART"
        fi
      done
    done
  fi
fi


if [ $do_nlproj -gt 0 ]; then
  #$EXECSTR $interactive run_nlproj.py --     -r $raw

  function nlproj {
    RS="run_nlproj.py -- $1"
    if [ $PREFIX_ALL_FEATS -gt 0 ]; then
      $EXECSTR $RS --prefix all                                                           
    fi
    if [ $PREFIXES_MAIN -gt 0 ]; then
      if [ $? -ne 0 ]; then
        echo "Somthing failed, exiting"
        exit 1
      fi
      $EXECSTR $RS --prefix LFPrel                  
      $EXECSTR $RS --prefix LFPrel_noself                  
      $EXECSTR $RS --prefix modLFP                                           
      $EXECSTR $RS --prefix modSrc                             
      $EXECSTR $RS --prefix allb_trem                             
      $EXECSTR $RS --prefix allb_beta       
      $EXECSTR $RS --prefix allb_gamma      
      $EXECSTR $RS --prefix allb_HFO      
    fi
    if [ $PREFIXES_AUX_SRC -gt 0 ]; then
      $EXECSTR $RS --prefix onlyMotorSrc   
      $EXECSTR $RS --prefix onlyRestSrc    
      $EXECSTR $RS --prefix onlyCBSrc      
    fi
    if [ $PREFIXES_AUX -gt 0 ]; then
      $EXECSTR $RS --prefix allnoHFO                 
      $EXECSTR $RS --prefix allb_trembeta   
      $EXECSTR $RS --prefix modLFPnoHFO 
      $EXECSTR $RS --prefix onlyTD            
      $EXECSTR $RS --prefix onlyFD                          
      $EXECSTR $RS --prefix conH                 
      $EXECSTR $RS --prefix onlyH                    
      $EXECSTR $RS --prefix onlyBpcorr                      
      $EXECSTR $RS --prefix onlyBpcorrNoHFO     
      $EXECSTR $RS --prefix onlyRbcorr                      
      $EXECSTR $RS --prefix onlyCon                         
      $EXECSTR $RS --prefix onlyConNoHFO        
    fi
    #$EXECSTR $RS --prefix all             
    #$EXECSTR $RS --prefix modLFP          
    #if [ $PREFIXES_AUX -gt 0 ]; then
    #  $EXECSTR $RS --prefix modSrc          
    #  $EXECSTR $RS --prefix allnoHFO             

    #  $EXECSTR $RS --prefix allb_trem       
    #  $EXECSTR $RS --prefix allb_beta       
    #  $EXECSTR $RS --prefix allb_gamma      
    #  $EXECSTR $RS --prefix allb_trembeta   

    #  $EXECSTR $RS --prefix onlyTD          
    #  $EXECSTR $RS --prefix onlyFD          
    #  $EXECSTR $RS --prefix onlyH           
    #  $EXECSTR $RS --prefix conH            
    #  $EXECSTR $RS --prefix modLFPnoHFO          
    #  $EXECSTR $RS --prefix onlyRbcorr      
    #  $EXECSTR $RS --prefix onlyCon         
    #  $EXECSTR $RS --prefix onlyBpcorrNoHFO 
    #  $EXECSTR $RS --prefix onlyConNoHFO    
    #  $EXECSTR $RS --prefix onlyBpcorr      
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
    RUNSTRING_COMMON="--subskip $SUBSKIP --dim_inp_nlproj $dim_inp_nlproj --show_plots $TSNE_PLOTS --load_nlproj $LOAD_TSNE $SUBDIR_STR"
  if [ $DO_TSNE_USING_INDIVID_ML -gt 0 ]; then
    NRAWS_ML=1
    RUNSTRING="--nrML $NRAWS_ML $RUNSTRING_COMMON"
    if [ $DO_RAW -gt 0 ]; then
      COMMON_PART="-r $raw $RUNSTRING"
      nlproj "$COMMON_PART"
    fi

    if [ $DO_RAW_COMPL -gt 0 ]; then
      COMMON_PART="-r $raw_compl $RUNSTRING"
      nlproj "$COMMON_PART"
    fi

    if [ $DO_RAW_BOTH -gt 0 ]; then
      COMMON_PART="-r $raws $RUNSTRING"
      nlproj "$COMMON_PART"
    fi
  fi

  if [ $DO_TSNE_USING_COMMON_ML -gt 0 ]; then
    #NRAWS_ML=2
    NRAWS_ML=${#raws_arr[*]}
    echo NRAWS_ML=$NRAWS_ML
    RUNSTRING="--nrML $NRAWS_ML $RUNSTRING_COMMON"
    if [ $DO_RAW -gt 0 ]; then
      COMMON_PART="-r $raw $RUNSTRING"
      nlproj "$COMMON_PART"
    fi

    if [ $DO_RAW_COMPL -gt 0 ]; then
      COMMON_PART="-r $raw_compl $RUNSTRING"
      nlproj "$COMMON_PART"
    fi

    if [ $DO_RAW_BOTH -gt 0 ]; then
      COMMON_PART="-r $raws $RUNSTRING"
      nlproj "$COMMON_PART"
    fi
  fi

fi




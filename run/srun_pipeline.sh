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
ML_SUBDIR=""


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
USE_AUX_IVAL_GROUPINGS=0

skip_XGB=0
skip_XGB_aux_int=0
# default val is 30, the smaller the longer the computation takes
max_XGB_step_nfeats=5

if [ $SAVE_RUNSTR_MODE -ne 0 ]; then
  RUN_CLASS_LAB_GRP_SEPARATE=1
else
  RUN_CLASS_LAB_GRP_SEPARATE=0
fi

#PREFIX_ALL_FEATS=1
#PREFIXES_MAIN=1
#PREFIXES_CROSS_MOD_AUX=1
#PREFIXES_AUX_SRC=1

######## common for ML and nlproj
#if [ -z ${var+x} ]; then echo "var is unset"; else echo "var is set to '$var'"; fi
            PREFIX_ALL_FEATS=0
               PREFIXES_MAIN=1
                PREFIXES_TMP=0
               PREFIXES_TMP2=0
               PREFIXES_TMP3=0
               PREFIXES_TMP4=0
      PREFIXES_CROSS_MOD_AUX=1
            PREFIXES_AUX_SRC=0
PREFIXES_CROSS_BPCORR_SUBMOD=0
PREFIXES_CROSS_BPCORR_SUBMOD_ORDBAND=0
PREFIXES_CROSS_BPCORR_SUBMOD_ORDBAND2=0
PREFIXES_CROSS_BPCORR_SUBMOD_ORDBAND3=0  # subset of ORDBAND2
PREFIXES_CROSS_RBCORR_SUBMOD=0
                PREFIXES_AUX=0
           PREFIX_SEARCH_LFP=0
           PREFIXES_BAND_NOH=0

#ALLFEATS_ADD_OPTS="--search_best_LFP LDA"
ALLFEATS_ADD_OPTS=""

#previously IPYTHON_SYNTAX
USE_IPYTHON=0
if [ $USE_IPYTHON -ne 0 ]; then
  DDASH="--"
else
  DDASH=""
fi


if [ $USE_AUX_IVAL_GROUPINGS -gt 0 ]; then
  GROUPINGS_TO_USE="$GROUPINGS_TO_USE merge_within_subj merge_within_medcond merge_within_task"
  INT_SETS_TO_USE="$INT_SETS_TO_USE subj_medcond_task subj_medcond subj"
fi

echo "GROUPINGS_TO_USE=$GROUPINGS_TO_USE"
echo "INT_SETS_TO_USE=$INT_SETS_TO_USE"
# on custer I canno set IFS for some reason. And bash has different version, so readarray won't work
#defIFS=$IFS
#IFS=','; echo .$IFS.
#echo .$IFS. 
#readarray -d : -t strarr <<< "$mainstr"
read -a int_sets_arr  <<< $INT_SETS_TO_USE
read -a groupings_arr <<< $GROUPINGS_TO_USE
#IFS=$defIFS
echo ${groupings_arr[0]} : ${groupings_arr[1]} 
#exit 0



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

if [ ${#GENFEATS_PARAM_FILE} -eq 0 ]; then
  GENFEATS_PARAM_FILE=genfeats_defparams.ini
fi
if [ ${#ML_PARAM_FILE} -eq 0 ]; then
  ML_PARAM_FILE=MI_defparams.ini
fi
if [ ${#NLPROJ_PARAM_FILE} -eq 0 ]; then
  NLPROJ_PARAM_FILE=nlproj_defparams.ini
fi

echo GENFEATS_PARAM_FILE=$GENFEATS_PARAM_FILE
echo ML_PARAM_FILE=$ML_PARAM_FILE

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
  BANDS_TREMOR=tremor                   
else
  BANDS_BETA=beta                  
  BANDS_GAMMA=gamma                   
  BANDS_HFO=HFO                   
  BANDS_TREMOR=tremor                   
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
  RUNSTRING_CUR=' run_genfeats.py $DDASH -r "$raws" --param_file $GENFEATS_PARAM_FILE $GENFEATS_INTERMED  $SUBDIR_STR --scale_data_combine_type $SCALE_DATA_COMBINE_TYPE'
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
    RS="run_ML.py $DDASH $1"
    #echo $RS --prefix all
    #echo RS= $RS

    # FAST TEST ONLY
    #$EXECSTR $RS --mods LFP                               --prefix modLFP            
    RUNSTRINGS=()

    if [ $PREFIX_ALL_FEATS -gt 0 ]; then
      # use everything (from main trem side)
      RUNSTRING_CUR=' $RS $ALLFEATS_ADD_OPTS --prefix all'
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi
    if [ $PREFIXES_BAND_NOH -gt 0 ]; then
      RUNSTRING_CUR=' $RS --fbands $BANDS_BETA  --feat_types con,rbcorr --prefix allb_beta_noH   '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --fbands $BANDS_GAMMA   --feat_types con,rbcorr  --prefix allb_gamma_noH  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --fbands $BANDS_TREMOR  --feat_types con,rbcorr  --prefix allb_tremor_noH '
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
      RUNSTRING_CUR=' $RS --feat_types H_act                     --prefix onlyH_act '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl        --prefix onlyH           '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --self_couplings_only 1 --mods msrc      --prefix modSrc_self  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --mods msrc         --prefix modSrc      '              
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --LFP_related_only 1 --cross_couplings_only 1 --prefix LFPrel_noself ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --fbands $BANDS_BETA  --feat_types con,rbcorr --prefix allb_beta_noH   '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --fbands $BANDS_GAMMA   --feat_types con,rbcorr  --prefix allb_gamma_noH  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --fbands $BANDS_TREMOR  --feat_types con,rbcorr  --prefix allb_tremor_noH '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      # most contribution comes from Hjorth if I include it
      #RUNSTRING_CUR=' $RS --fbands $BANDS_BETA                     --prefix allb_beta            '
      #RUNSTRINGS+=("$RUNSTRING_CUR")
      #RUNSTRING_CUR=' $RS --fbands $BANDS_GAMMA                     --prefix allb_gamma           '
      #RUNSTRINGS+=("$RUNSTRING_CUR")
      #RUNSTRING_CUR=' $RS --fbands $BANDS_TREMOR                     --prefix allb_tremor          '
      #RUNSTRINGS+=("$RUNSTRING_CUR")
    fi
    if [ $PREFIXES_TMP -gt 0 ]; then
      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl --parcel_group_names !Cerebellum  --prefix onlyH_noCB '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl --parcel_group_names !OccipitalMid  --prefix onlyH_noOccipitalMid '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl --parcel_group_names !FrontalMed  --prefix onlyH_noFrontalMed '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl --parcel_group_names !FrontalSup  --prefix onlyH_noFrontalSup '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl --parcel_group_names !TemporalMid  --prefix onlyH_noTemporalMid '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl --parcel_group_names !Sensorimotor  --prefix onlyH_noSensorimotor '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl --mods msrc --prefix onlyH_noLFP '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi

    if [ $PREFIXES_TMP2 -gt 0 ]; then
      RUNSTRING_CUR=' $RS --feat_types H_act --prefix onlyH_act '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types H_mob --prefix onlyH_mob '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types H_compl --prefix onlyH_compl '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob --prefix onlyH_nocompl '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types H_mob,H_compl --prefix onlyH_noact '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types H_act,H_compl --prefix onlyH_nomob '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi

    if [ $PREFIXES_TMP3 -gt 0 ]; then
      RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !Cerebellum  --prefix onlyH_act_noCB '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !OccipitalMid  --prefix onlyH_act_noOccipitalMid '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !FrontalMed  --prefix onlyH_act_noFrontalMed '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !FrontalSup  --prefix onlyH_act_noFrontalSup '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !TemporalMid  --prefix onlyH_act_noTemporalMid '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !Sensorimotor  --prefix onlyH_act_noSensorimotor '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --feat_types H_act --mods msrc --prefix onlyH_act_noLFP '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi

    if [ $PREFIXES_TMP4 -gt 0 ]; then
      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !Cerebellum  --prefix onlyH_act_noCB '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !ParietalSup  --prefix onlyH_act_noParietalSup '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFS '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPS '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !ParietalInf,!ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPSnoPI '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !FrontalMed,!ParietalInf,!ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPSnoPInoFM '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !OccipitalInf,!FrontalMed,!ParietalInf,!ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPSnoPInoFMnoOI '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !OccipitalMid,!OccipitalInf,!FrontalMed,!ParietalInf,!ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPSnoPInoFMnoOInoOM '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !Angular,!OccipitalMid,!OccipitalInf,!FrontalMed,!ParietalInf,!ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPSnoPInoFMnoOInoOMnoA '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #--------
      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !TemporalInf,!Angular,!OccipitalMid,!OccipitalInf,!FrontalMed,!ParietalInf,!ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPSnoPInoFMnoOInoOMnoAnoTI '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !Sensorimotor,!TemporalInf,!Angular,!OccipitalMid,!OccipitalInf,!FrontalMed,!ParietalInf,!ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPSnoPInoFMnoOInoOMnoAnoTInoSM '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !Sensorimotor --prefix onlyH_act_noSM '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !TemporalSup,!Sensorimotor,!TemporalInf,!Angular,!OccipitalMid,!OccipitalInf,!FrontalMed,!ParietalInf,!ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPSnoPInoFMnoOInoOMnoAnoTInoSMnoTS '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names !FrontalInf,!TemporalSup,!Sensorimotor,!TemporalInf,!Angular,!OccipitalMid,!OccipitalInf,!FrontalMed,!ParietalInf,!ParietalSup,!FrontalSup,!Cerebellum  --prefix onlyH_act_noCBnoFSnoPSnoPInoFMnoOInoOMnoAnoTInoSMnoTSnoFI '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      ##--------------

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names Cerebellum  --prefix onlyH_act_CB '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names Sensorimotor,Cerebellum  --prefix onlyH_act_CBySM '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names FrontalSup,Cerebellum  --prefix onlyH_act_CByFS '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names ParietalSup,FrontalSup,Cerebellum  --prefix onlyH_act_CByFSyPS '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names ParietalInf,ParietalSup,FrontalSup,Cerebellum  --prefix onlyH_act_CByFSyPSyPI '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names FrontalMed,ParietalInf,ParietalSup,FrontalSup,Cerebellum  --prefix onlyH_act_CByFSyPSyPIyFM '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names Sensorimotor  --prefix onlyH_act_SM '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names Sensorimotor,OccipitalInf  --prefix onlyH_act_SMyOI '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names Sensorimotor,OccipitalInf,FrontalSup  --prefix onlyH_act_SMyOIyFS '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names Sensorimotor,OccipitalInf,FrontalSup,FrontalInf  --prefix onlyH_act_SMyOIyFSyFI '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      #RUNSTRING_CUR=' $RS --feat_types H_act --parcel_group_names Sensorimotor,OccipitalInf,FrontalSup,FrontalInf,TemporalMid  --prefix onlyH_act_SMyOIyFSyFIyTM '        
      #RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --mods msrc --feat_types H_act --parcel_group_names Sensorimotor  --prefix onlyH_act_SM_noLFP '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --mods msrc --feat_types H_act --parcel_group_names Sensorimotor,OccipitalInf  --prefix onlyH_act_SMyOI_noLFP '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --mods msrc --feat_types H_act --parcel_group_names Sensorimotor,OccipitalInf,FrontalSup  --prefix onlyH_act_SMyOIyFS_noLFP '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --mods msrc --feat_types H_act --parcel_group_names Sensorimotor,OccipitalInf,FrontalSup,FrontalInf  --prefix onlyH_act_SMyOIyFSyFI_noLFP '        
      RUNSTRINGS+=("$RUNSTRING_CUR")

      RUNSTRING_CUR=' $RS --mods msrc --feat_types H_act --parcel_group_names Sensorimotor,OccipitalInf,FrontalSup,FrontalInf,TemporalMid  --prefix onlyH_act_SMyOIyFSyFIyTM_noLFP '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi


    if [ $PREFIXES_AUX_SRC -gt 0 ]; then
      RUNSTRING_CUR=' $RS --parcel_group_names Sensorimotor       --prefix onlyMotorSrc  '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --parcel_group_names !Sensorimotor    --prefix onlyRestSrc   '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --parcel_group_names Cerebellum         --prefix onlyCBSrc     '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi
    if [ $PREFIXES_CROSS_MOD_AUX -ne 0 ]; then 
      CROSS_MOD="--LFP_related_only 1 --cross_couplings_only 1"
      RUNSTRING_CUR=' $RS $CROSS_MOD --feat_types con     --prefix LFPrel_noself_onlyCon     '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $CROSS_MOD --feat_types rbcorr  --prefix LFPrel_noself_onlyRbcorr  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $CROSS_MOD --feat_types bpcorr  --prefix LFPrel_noself_onlyBpcorr  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi 
    if [ $PREFIX_SEARCH_LFP -ne 0 ]; then
      RUNSTRING_CUR=' $RS --mods LFP        --prefix modLFP               '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --LFP_related_only 1 --cross_couplings_only 1   --prefix LFPrel_noself '       
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS --feat_types H_act,H_mob,H_compl   --prefix onlyH           '        
      RUNSTRINGS+=("$RUNSTRING_CUR")
      CROSS_MOD="--LFP_related_only 1 --cross_couplings_only 1"
      RUNSTRING_CUR=' $RS $CROSS_MOD --feat_types con      --prefix LFPrel_noself_onlyCon     '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $CROSS_MOD --feat_types rbcorr   --prefix LFPrel_noself_onlyRbcorr  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $CROSS_MOD --feat_types bpcorr   --prefix LFPrel_noself_onlyBpcorr  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi
    if [ $PREFIXES_CROSS_BPCORR_SUBMOD -ne 0 ]; then 
      BPCORR_SUBMOD="--LFP_related_only 1 --cross_couplings_only 1 --feat_types bpcorr"
      #########
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands tremor,$BANDS_BETA --prefix LFPrel_noself_onlyBpcorr_trembeta  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands tremor,$BANDS_GAMMA --prefix LFPrel_noself_onlyBpcorr_tremgamma  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands $BANDS_BETA,$BANDS_GAMMA --prefix LFPrel_noself_onlyBpcorr_betagamma  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      ########################################
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --parcel_group_names $PARCEL_TYPES_MOTOR       --prefix LFPrel_noself_onlyBpcorr_onlyMotorSrc  '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --parcel_group_names $PARCEL_TYPES_NONMOTOR    --prefix LFPrel_noself_onlyBpcorr_onlyRestSrc   '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --parcel_types $PARCEL_TYPES_CB                --prefix LFPrel_noself_onlyBpcorr_onlyCBSrc     '   
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi 

    if [ $PREFIXES_CROSS_BPCORR_SUBMOD_ORDBAND -ne 0 ]; then 
      BPCORR_SUBMOD="--LFP_related_only 1 --cross_couplings_only 1 --feat_types bpcorr"
      #
      #cross_freqmod_
      #########
 #--fbands_mod1 msrc:beta --fbands_mod2 LFP:gamma,tremor
      # NOTE THAT I CANNOT USE LOCAL VARIABLE HERE! ONLY THE LAST WILL BE REMEMBERED
      b_msrc="$BANDS_GAMMA"; b_LFP="$BANDS_BETA";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:$BANDS_GAMMA --fbands_mod2 LFP:$BANDS_BETA --prefix cross_freqmod_$BANDS_GAMMA:$BANDS_BETA ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="tremor"; b_LFP="$BANDS_BETA";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:tremor --fbands_mod2 LFP:$BANDS_BETA --prefix cross_freqmod_tremor:$BANDS_BETA ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="$BANDS_GAMMA"; b_LFP="tremor";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:$BANDS_GAMMA --fbands_mod2 LFP:tremor --prefix cross_freqmod_$BANDS_GAMMA:tremor ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="$BANDS_BETA"; b_LFP="tremor";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:$BANDS_BETA --fbands_mod2 LFP:tremor --prefix cross_freqmod_$BANDS_BETA:tremor ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="tremor"; b_LFP="$BANDS_GAMMA";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:tremor --fbands_mod2 LFP:$BANDS_GAMMA --prefix cross_freqmod_tremor:$BANDS_GAMMA ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="$BANDS_BETA"; b_LFP="$BANDS_GAMMA";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:$BANDS_BETA --fbands_mod2 LFP:$BANDS_GAMMA --prefix cross_freqmod_$BANDS_BETA:$BANDS_GAMMA ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="tremor"; b_LFP="$BANDS_HFO";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:tremor --fbands_mod2 LFP:$BANDS_HFO --prefix cross_freqmod_tremor:$BANDS_HFO ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="$BANDS_BETA"; b_LFP="$BANDS_HFO";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:$BANDS_BETA --fbands_mod2 LFP:$BANDS_HFO --prefix cross_freqmod_$BANDS_BETA:$BANDS_HFO ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="$BANDS_GAMMA"; b_LFP="$BANDS_HFO";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:$BANDS_GAMMA --fbands_mod2 LFP:$BANDS_HFO --prefix cross_freqmod_$BANDS_GAMMA:$BANDS_HFO ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi 

    if [ $PREFIXES_CROSS_BPCORR_SUBMOD_ORDBAND2 -ne 0 ]; then 
      BPCORR_SUBMOD="--LFP_related_only 1 --cross_couplings_only 1 --feat_types bpcorr"
      #########
 #--fbands_mod1 msrc:$BANDS_BETA --fbands_mod2 LFP:$BANDS_GAMMA,tremor
      b_msrc="tremor,$BANDS_GAMMA"; b_LFP="$BANDS_BETA";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:tremor,$BANDS_GAMMA --fbands_mod2 LFP:$BANDS_BETA --prefix cross_freqmod_tremor,$BANDS_GAMMA:$BANDS_BETA ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="tremor,$BANDS_BETA"; b_LFP="$BANDS_GAMMA";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:tremor,$BANDS_BETA --fbands_mod2 LFP:$BANDS_GAMMA --prefix cross_freqmod_tremor,$BANDS_BETA:$BANDS_GAMMA ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="$BANDS_BETA,$BANDS_GAMMA"; b_LFP="tremor";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:$BANDS_BETA,$BANDS_GAMMA --fbands_mod2 LFP:tremor --prefix cross_freqmod_$BANDS_BETA,$BANDS_GAMMA:tremor ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="tremor,$BANDS_BETA"; b_LFP="$BANDS_HFO";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:tremor,$BANDS_BETA --fbands_mod2 LFP:$BANDS_HFO --prefix cross_freqmod_tremor,$BANDS_BETA:$BANDS_HFO ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="tremor,$BANDS_GAMMA"; b_LFP="$BANDS_HFO";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:tremor,$BANDS_GAMMA --fbands_mod2 LFP:$BANDS_HFO --prefix cross_freqmod_tremor,$BANDS_GAMMA:$BANDS_HFO ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="$BANDS_BETA,$BANDS_GAMMA"; b_LFP="$BANDS_HFO";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:$BANDS_BETA,$BANDS_GAMMA --fbands_mod2 LFP:$BANDS_HFO --prefix cross_freqmod_$BANDS_BETA,$BANDS_GAMMA:$BANDS_HFO ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi 
    #s/b_msrc="\([a-zA-Z,]\+\)"; b_LFP="\([a-zA-Z,]\+\)";/\0\r      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:\1 --fbands_mod2 LFP:\2 --prefix cross_freqmod_\1:\2 '/gc

    if [ $PREFIXES_CROSS_BPCORR_SUBMOD_ORDBAND3 -ne 0 ]; then 
      BPCORR_SUBMOD="--LFP_related_only 1 --cross_couplings_only 1 --feat_types bpcorr"
      #########
 #--fbands_mod1 msrc:$BANDS_BETA --fbands_mod2 LFP:$BANDS_GAMMA,tremor
      b_msrc="tremor,$BANDS_GAMMA"; b_LFP="$BANDS_BETA";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:tremor,$BANDS_GAMMA --fbands_mod2 LFP:$BANDS_BETA --prefix cross_freqmod_tremor,$BANDS_GAMMA:$BANDS_BETA ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="tremor,$BANDS_BETA"; b_LFP="$BANDS_HFO";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:tremor,$BANDS_BETA --fbands_mod2 LFP:$BANDS_HFO --prefix cross_freqmod_tremor,$BANDS_BETA:$BANDS_HFO ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")

      b_msrc="$BANDS_BETA,$BANDS_GAMMA"; b_LFP="$BANDS_HFO";
      RUNSTRING_CUR=' $RS $BPCORR_SUBMOD --fbands_mod1 msrc:$BANDS_BETA,$BANDS_GAMMA --fbands_mod2 LFP:$BANDS_HFO --prefix cross_freqmod_$BANDS_BETA,$BANDS_GAMMA:$BANDS_HFO ' 
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi 

    

    if [ $PREFIXES_CROSS_RBCORR_SUBMOD -ne 0 ]; then 
      RBCORR_SUBMOD="--LFP_related_only 1 --cross_couplings_only 1 --feat_types rbcorr"
      #########
      RUNSTRING_CUR=' $RS $RBCORR_SUBMOD --fbands $BANDS_BETA --prefix LFPrel_noself_onlyRbcorr_beta  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $RBCORR_SUBMOD --fbands $BANDS_GAMMA --prefix LFPrel_noself_onlyRbcorr_gamma  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $RBCORR_SUBMOD --fbands tremor --prefix LFPrel_noself_onlyRbcorr_trem  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $RBCORR_SUBMOD --fbands tremor,$BANDS_BETA --prefix LFPrel_noself_onlyRbcorr_trembeta  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $RBCORR_SUBMOD --fbands tremor,$BANDS_GAMMA --prefix LFPrel_noself_onlyRbcorr_tremgamma  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
      RUNSTRING_CUR=' $RS $RBCORR_SUBMOD --fbands $BANDS_BETA,$BANDS_GAMMA --prefix LFPrel_noself_onlyRbcorr_betagamma  '
      RUNSTRINGS+=("$RUNSTRING_CUR")
    fi 
    if [ $PREFIXES_AUX -gt 0 ]; then
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
  else
    INPUT_SUBDIR_STR=""
  fi

  if [ ${#ML_SUBDIR} -gt 0 ]; then
    OUTPUT_SUBDIR_STR="--output_subdir $ML_SUBDIR"
  else
    OUTPUT_SUBDIR_STR=""
  fi

  SUBDIR_STR="$INPUT_SUBDIR_STR $OUTPUT_SUBDIR_STR"
  #COMMON_ALL="--pcexpl $DESIRED_PCA_EXPLAIN --discard $DESIRED_DISCARD --show_plots $ML_PLOTS --sources_type $SOURCES_TYPE --bands_type $BANDS_TYPE --src_grouping $SRC_GROUPING --src_grouping_fn $SRC_GROUPING_FN --skip_XGB $skip_XGB --skip_XGB_aux_int $skip_XGB_aux_int --max_XGB_step_nfeats $max_XGB_step_nfeats --subskip_fit $SUBSKIP_ML_FIT --LFPchan $LFP_CHAN_TO_USE --heavy_fit_red_featset $HEAVY_FIT_REDUCED_FEATSET $SUBDIR_STR --plot_types $ML_PLOT_TYPES --load_only $ML_LOAD_ONLY --param_file $ML_PARAM_FILE"
  COMMON_ALL="--param_file $ML_PARAM_FILE $SUBDIR_STR"
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
        cd ..
        OUT=$(python3 -c "$S")
        # first line is hostname so take the last line only
        OUT=`echo "$OUT" | tail -n1`
        cd run
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
    RS="run_nlproj.py $DDASH $1"
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




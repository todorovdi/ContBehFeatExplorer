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

#SOURCES_TYPE=""
SOURCES_TYPE="parcel_aal"

DESIRED_PCA_EXPLAIN=0.95
DESIRED_DISCARD=0.01
PCA_PLOTS=0

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

PREFIX_ALL_FEATS=1
PREFIXES_MAIN=1
PREFIXES_AUX=1
PREFIXES_AUX_SRC=1
LFP_CHAN_TO_USE="all"
#LFP_CHAN_TO_USE="main"
#BANDS_TYPE="fine"
BANDS_TYPE="crude"
FEAT_TYPES_TO_USE="con,H_act,H_mob,H_compl,rbcorr,bpcorr"    # for run_genfeats only
SRC_GROUPING=0      # index of grouping inside the file
#SRC_GROUPING_FN=9
SRC_GROUPING_FN=10

USE_AUX_GROUPINGS=1

#--groupings_to_use
#GROUPINGS_TO_USE="merge_all_not_trem,merge_movements,merge_nothing"
GROUPINGS_TO_USE="merge_nothing"

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
SAVE_TFR=0
SAVE_CSD=0
LOAD_TFR=1
LOAD_CSD=1
SAVE_RBCORR=1
SAVE_BPCORR=1
 
PARCEL_TYPES_CB="Cerebellum"
#PARCEL_TYPES_NONMOTOR="rest"
PARCEL_TYPES_NONMOTOR="not_motor-related"
PARCEL_TYPES_MOTOR="motor-related" 

if [[ $BANDS_TYPE == "fine" ]]; then
  BANDS_BETA=low_beta,high_beta                  
  BANDS_GAMMA=low_gamma,high_gamma                   
else
  BANDS_BETA=beta                  
  BANDS_GAMMA=gamma                   
fi

if [ $do_genfeats -gt 0 ]; then
  GENFEATS_INTERMED="--load_TFR $LOAD_TFR --load_CSD $LOAD_CSD --save_TFR $SAVE_TFR --save_CSD $SAVE_CSD --save_rbcorr $SAVE_RBCORR --save_bpcorr $SAVE_BPCORR"
  ipython3 $interactive run_genfeats.py -- -r "$raws" --bands "$BANDS_TYPE" --sources_type "$SOURCES_TYPE" --feat_types "$FEAT_TYPES_TO_USE" --crop "$CROP" --src_grouping "$SRC_GROUPING" --src_grouping_fn $SRC_GROUPING_FN --Kalman_smooth $KALMAN $GENFEATS_INTERMED
fi

if [ $do_PCA -gt 0 ]; then

  #ipython3 $interactive run_PCA.py  -- -r $raw,$raw_compl

  function PCA { 
    RS="$interactive run_PCA.py -- $1"
    #echo $RS --prefix all
    #echo RS= $RS
    # use everything (from main trem side)

    if [ $PREFIX_ALL_FEATS -gt 0 ]; then
      ipython3 $RS --prefix all  --search_best_LFP 0
    fi
    if [ $PREFIXES_MAIN -gt 0 ]; then
      # searching for best LFP with all features takes too much time and makes little sense because 
      # there is too much redundancy
      if [ $? -ne 0 ]; then  #checking exit code
        echo "trying to run with all features resulted in an error, exiting"
        exit 1
      fi
      ipython3 $RS --mods LFP                    --prefix modLFP            
      ipython3 $RS --LFP_related_only 1          --prefix LFPrel                                    
      ipython3 $RS --mods msrc                   --prefix modSrc                             
      ipython3 $RS --fbands $BANDS_BETA          --prefix allb_beta            
      ipython3 $RS --fbands $BANDS_GAMMA         --prefix allb_gamma           
      ipython3 $RS --fbands tremor               --prefix allb_trem                                  
    fi
    if [ $PREFIXES_AUX_SRC -gt 0 ]; then
      ipython3 $RS --prefix onlyMotorSrc       --parcel_group_names $PARCEL_TYPES_MOTOR 
      ipython3 $RS --prefix onlyRestSrc        --parcel_group_names $PARCEL_TYPES_NONMOTOR 
      ipython3 $RS --prefix onlyCBSrc          --parcel_types $PARCEL_TYPES_CB 
    fi
    if [ $PREFIXES_AUX -gt 0 ]; then
      # TODO: add   LFP_related_only  but with exclusion of self-couplings
      ipython3 $RS --prefix allnoHFO        --use_HFO 0                               
      #ipython3 $RS --prefix allb_trem       --fbands tremor                          
      #ipython3 $RS --prefix allb_gamma      --fbands low_gamma,high_gamma            
      ipython3 $RS --prefix allb_trembeta   --fbands tremor,$BANDS_BETA      
      ipython3 $RS --mods LFP               --prefix modLFPnoHFO --use_HFO 0          
      ipython3 $RS --feat_types H_act,H_mob,H_compl,rbcorr --prefix onlyTD            
      ipython3 $RS --feat_types con,bpcorr   --prefix onlyFD                          
      ipython3 $RS --feat_types con,H_act,H_mob,H_compl --prefix conH                 
      ipython3 $RS --feat_types H_act,H_mob,H_compl --prefix onlyH                    
      ipython3 $RS --feat_types bpcorr       --prefix onlyBpcorr                      
      ipython3 $RS --feat_types bpcorr       --use_HFO 0 --prefix onlyBpcorrNoHFO     
      ipython3 $RS --feat_types rbcorr       --prefix onlyRbcorr                      
      ipython3 $RS --feat_types con          --prefix onlyCon                         
      ipython3 $RS --feat_types con          --use_HFO 0 --prefix onlyConNoHFO        
    fi
  }

  COMMON_ALL="--pcexpl $DESIRED_PCA_EXPLAIN --discard $DESIRED_DISCARD --show_plots $PCA_PLOTS --sources_type $SOURCES_TYPE --single_core $single_core --bands_type $BANDS_TYPE --src_grouping $SRC_GROUPING --src_grouping_fn $SRC_GROUPING_FN --groupings_to_use=$GROUPINGS_TO_USE --int_types_to_use=$INT_TYPES_TO_USE --skip_XGB $skip_XGB --subskip_fit $SUBSKIP_ML_FIT --LFPchan $LFP_CHAN_TO_USE --heavy_fit_red_featset $HEAVY_FIT_REDUCED_FEATSET"

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
    if [ $PREFIXES_MAIN -gt 0 ]; then
      ipython3 $RS --prefix all                                                           
      if [ $? -ne 0 ]; then
        exit 1
      fi
      ipython3 $RS --LFP_related_only 1     --prefix LFPrel                  
      ipython3 $RS --mods LFP               --prefix modLFP                                           
      ipython3 $RS --fbands $BANDS_BETA     --prefix allb_beta       
      ipython3 $RS --mods msrc              --prefix modSrc                             
      ipython3 $RS --fbands tremor          --prefix allb_trem                             
      ipython3 $RS --fbands $BANDS_GAMMA    --prefix allb_gamma      
    fi
    if [ $PREFIXES_AUX_SRC -gt 0 ]; then
      ipython3 $RS --prefix onlyMotorSrc       --parcel_types $PARCEL_TYPES_MOTOR 
      ipython3 $RS --prefix onlyRestSrc        --parcel_types $PARCEL_TYPES_NONMOTOR 
      ipython3 $RS --prefix onlyCBSrc          --parcel_types $PARCEL_TYPES_CB 
    fi
    if [ $PREFIXES_AUX -gt 0 ]; then
      ipython3 $RS --prefix allnoHFO        --use_HFO 0                               
      #ipython3 $RS --prefix allb_trem       --fbands tremor                          
      #ipython3 $RS --prefix allb_gamma      --fbands low_gamma,high_gamma            
      ipython3 $RS --prefix allb_trembeta   --fbands tremor,$BANDS_BETA      
      ipython3 $RS --mods LFP               --prefix modLFPnoHFO --use_HFO 0          
      ipython3 $RS --feat_types H_act,H_mob,H_compl,rbcorr --prefix onlyTD            
      ipython3 $RS --feat_types con,bpcorr   --prefix onlyFD                          
      ipython3 $RS --feat_types con,H_act,H_mob,H_compl --prefix conH                 
      ipython3 $RS --feat_types H_act,H_mob,H_compl --prefix onlyH                    
      ipython3 $RS --feat_types bpcorr       --prefix onlyBpcorr                      
      ipython3 $RS --feat_types bpcorr       --use_HFO 0 --prefix onlyBpcorrNoHFO     
      ipython3 $RS --feat_types rbcorr       --prefix onlyRbcorr                      
      ipython3 $RS --feat_types con          --prefix onlyCon                         
      ipython3 $RS --feat_types con          --use_HFO 0 --prefix onlyConNoHFO        
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


    RUNSTRING_COMMON="--subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE"
  if [ $DO_TSNE_USING_INDIVID_PCA -gt 0 ]; then
    NRAWS_PCA=1
    RUNSTRING="--nrPCA $NRAWS_PCA $RUNSTRING_COMMON"
    if [ $DO_RAW -gt 0 ]; then
      raws=$raw
      COMMON_PART="-r $raws $RUNSTRING"
      tSNE "$COMMON_PART"
    fi

    if [ $DO_RAW_COMPL -gt 0 ]; then
      raws=$raw_compl
      COMMON_PART="-r $raws $RUNSTRING"
      tSNE "$COMMON_PART"
    fi

    if [ $DO_RAW_BOTH -gt 0 ]; then
      raws=$raws
      COMMON_PART="-r $raws $RUNSTRING"
      tSNE "$COMMON_PART"
    fi
  fi

  if [ $DO_TSNE_USING_COMMON_PCA -gt 0 ]; then
    NRAWS_PCA=2
    RUNSTRING="--nrPCA $NRAWS_PCA $RUNSTRING_COMMON"
    if [ $DO_RAW -gt 0 ]; then
      raws=$raw
      COMMON_PART="-r $raws $RUNSTRING"
      tSNE "$COMMON_PART"
    fi

    if [ $DO_RAW_COMPL -gt 0 ]; then
      raws=$raw_compl
      COMMON_PART="-r $raws $RUNSTRING"
      tSNE "$COMMON_PART"
    fi

    if [ $DO_RAW_BOTH -gt 0 ]; then
      raws=$raws
      COMMON_PART="-r $raws $RUNSTRING"
      tSNE "$COMMON_PART"
    fi
  fi

fi




#!/bin/bash

do_genfeats=$1
do_PCA=$2
do_tSNE=$3
shift
shift 
shift

LOAD_TSNE=0  #allows to skip those that were already computed

# parse arguments
POSITIONAL=()
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
echo raw=$raw,raw_compl=$raw_compl

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

#SOURCES_TYPE=""
SOURCES_TYPE="parcel_aal"

DESIRED_PCA_EXPLAIN=0.95
DESIRED_DISCARD=0.01
PCA_PLOTS=1

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

DO_RAW_BOTH_PCA=1
DO_RAW_COMPL_PCA=0
DO_RAW_PCA=0

DO_RAW=0
DO_RAW_COMPL=0
DO_RAW_BOTH=1

PREFIXES_AUX=0
#FEAT_BANDS="fine"
FEAT_BANDS="crude"
FEAT_TYPES_TO_USE="con,H_act,H_mob,H_compl"

if [ $do_genfeats -gt 0 ]; then
  ipython3 $interactive run_genfeats.py -- -r $raw,$raw_compl --bands $FEAT_BANDS --sources_type $SOURCES_TYPE --feat_types FEAT_TYPES_TO_USE
fi

if [ $do_PCA -gt 0 ]; then

  #ipython3 $interactive run_PCA.py  -- -r $raw,$raw_compl

  function PCA { 
    RS="$interactive run_PCA.py -- $1 --single_core $single_core"
    # use everything (from main trem side)
    ipython3 $RS --prefix all                                                           
    if [$? -ne 0 ]; then
      exit 1
    fi
    ipython3 $RS --mods LFP --prefix modLFP                                           
    ipython3 $RS --prefix allb_beta       --fbands low_beta,high_beta                 
    ipython3 $RS --mods msrc              --prefix modSrc                             
    ipython3 $RS --prefix allb_trem       --fbands tremor                             
    ipython3 $RS --prefix allb_gamma      --fbands low_gamma,high_gamma               
    if [ $PREFIXES_AUX -gt 0 ]; then
      ipython3 $RS --prefix allnoHFO        --use_HFO 0                               
      #ipython3 $RS --prefix allb_trem       --fbands tremor                          
      #ipython3 $RS --prefix allb_gamma      --fbands low_gamma,high_gamma            
      ipython3 $RS --prefix allb_trembeta   --fbands tremor,low_gamma,high_gamma      
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

  if [ $DO_RAW_BOTH_PCA -gt 0 ]; then
    COMMON_PART="-r $raw,$raw_compl --pcexpl $DESIRED_PCA_EXPLAIN --discard $DESIRED_DISCARD --show_plots $PCA_PLOTS --sources_type $SOURCES_TYPE"
    #./subrun_pipeline_PCA.sh
    PCA "$COMMON_PART"
  fi

  if [ $DO_RAW_PCA -gt 0 ]; then
    COMMON_PART="-r $raw --pcexpl $DESIRED_PCA_EXPLAIN --discard $DESIRED_DISCARD --show_plots $PCA_PLOTS --sources_type $SOURCES_TYPE"
    #./subrun_pipeline_PCA.sh
    PCA "$COMMON_PART"
  fi

  if [ $DO_RAW_COMPL_PCA -gt 0 ]; then
    COMMON_PART="-r $raw_compl --pcexpl $DESIRED_PCA_EXPLAIN --discard $DESIRED_DISCARD --show_plots $PCA_PLOTS --sources_type $SOURCES_TYPE"
    #./subrun_pipeline_PCA.sh
    PCA "$COMMON_PART"
  fi

fi

if [ $do_tSNE -gt 0 ]; then
  #ipython3 $interactive run_tSNE.py --     -r $raw
 


  function tSNE {
    ipython3 $interactive run_tSNE.py -- $1 --prefix all             
    ipython3 $interactive run_tSNE.py -- $1 --prefix modLFP          
    if [ $PREFIXES_AUX -gt 0 ]; then
      ipython3 $interactive run_tSNE.py -- $1 --prefix modSrc          
      ipython3 $interactive run_tSNE.py -- $1 --prefix allnoHFO             

      ipython3 $interactive run_tSNE.py -- $1 --prefix allb_trem       
      ipython3 $interactive run_tSNE.py -- $1 --prefix allb_beta       
      ipython3 $interactive run_tSNE.py -- $1 --prefix allb_gamma      
      ipython3 $interactive run_tSNE.py -- $1 --prefix allb_trembeta   

      ipython3 $interactive run_tSNE.py -- $1 --prefix onlyTD          
      ipython3 $interactive run_tSNE.py -- $1 --prefix onlyFD          
      ipython3 $interactive run_tSNE.py -- $1 --prefix onlyH           
      ipython3 $interactive run_tSNE.py -- $1 --prefix conH            
      ipython3 $interactive run_tSNE.py -- $1 --prefix modLFPnoHFO          
      ipython3 $interactive run_tSNE.py -- $1 --prefix onlyRbcorr      
      ipython3 $interactive run_tSNE.py -- $1 --prefix onlyCon         
      ipython3 $interactive run_tSNE.py -- $1 --prefix onlyBpcorrNoHFO 
      ipython3 $interactive run_tSNE.py -- $1 --prefix onlyConNoHFO    
      ipython3 $interactive run_tSNE.py -- $1 --prefix onlyBpcorr      
    fi
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
      raws=$raw,$raw_compl
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
      raws=$raw,$raw_compl
      COMMON_PART="-r $raws $RUNSTRING"
      tSNE "$COMMON_PART"
    fi
  fi

fi




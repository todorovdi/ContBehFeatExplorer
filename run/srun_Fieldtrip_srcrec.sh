#!/bin/bash
# takes one argument
# this script should be called from the root code directory, NOT from 'run' subdirectory (otherwise matlab paths setting is a nightmare)

# in bash Inside single quotes everything is preserved literally, without exception.
#
# in matlab single quotes define a character vector with size 1xN, where N is
# the number of characters between the quotes. double quotes define a string
# array with size 1x1 (i.e. scalar, regardless of how many characters it has).
# You can think of string arrays as a kind of container for character vectors.

echo Starting `basename "$0"`

# it is important that here we have variable names not intersecting with those in srun_resave
#raws_comp
rawcur=$1

MERGE_RAWS=0      # actually not implemented yet when = 1, I need to merge in Fieldtrip and then de-merge for saving

if [ -z ${INPUT_SUBDIR+x} ]; then
  INPUT_SUBDIR=""
fi
if [ -z ${OUTPUT_SUBDIR+x} ]; then
  OUTPUT_SUBDIR=""
fi

DO_SOURCE_RECONSTRUCTION=1

if [ -z $MIN_DURATION_QUIET_ALLOWED ]; then
  echo "MIN_DURATION_QUIET_ALLOWED not defined, exiting"
  exit 1
fi

if [ -z $SRCREC_COVMAT_REST_ONLY ]; then
  echo "SRCREC_COVMAT_REST_ONLY not defined, exiting"
  exit 1
fi



CODERUN=$OSCBAGDIS_DATAPROC_CODE/run

ROI_TYPES='"'parcel_aal_surf'"'
#ROI_TYPES='"'HirschPt2011,2013'","'Thalamus'"'
#ROI_TYPES='"'parcel_aal_surf'","'Thalamus_L'","'Thalamus_R'"' 
#ROI_TYPES='"'HirschPt2011,2013'"'
#ROI_TYPES='"'Thalamus_L'","'Thalamus_R'"'

# this is a hack to check presense of the variable truly, not just whether it is zero  [ -z ${vardefstr+x} ]. 
# If it was present, adding x would lengthten the string. If not, you get nothing 
if [ -z ${vardefstr+x} ]; then
  vardefstr=""
fi
if [ -z ${MATLAB_SCRIPT_PARAMS+x} ]; then
  echo "srun_Fieldtrip_srcrec.sh   Empty MATLAB_SCRIPT_PARAMS, setting here"
  MATLAB_SCRIPT_PARAMS=""
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'input_rawname_type = ["resample", "notch", "highpass"];'
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'roi = {'$ROI_TYPES'};'
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS"use_data_afterICA=0;"
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'input_subdir="'$INPUT_SUBDIR'";'
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'output_subdir="'$OUTPUT_SUBDIR'";'
fi

echo "MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS"

if [ $SRCREC_COVMAT_REST_ONLY -ne 0 ]; then
  s="--ann_types $MEG_artif_types,beh_states"
else
  s="--ann_types $MEG_artif_types"
fi


# if just a single raw to process
if [ $MERGE_RAWS -eq 0 ]; then
  python $CODERUN/run_collect_artifacts.py -r $rawcur --min_dur $MIN_DURATION_QUIET_ALLOWED $s
  EC=$?
  if [ $EC -ne 0 ]; then
    echo "srun_Fieldtrip_srcrec.sh: run_collect_artifacts exit code $EC, exiting"
    exit $EC
  fi

  vardefstr=$MATLAB_SCRIPT_PARAMS'rawnames=["'$rawcur'"];'
  echo "vardefstr=$vardefstr"
  #exit 0
  #matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit"
  if [ $DO_SOURCE_RECONSTRUCTION -ne 0 ]; then
    matlab -nodisplay -nosplash -r "$vardefstr run(\"$OSCBAGDIS_DATAPROC_CODE/MEGsource_reconstruct_multiraw\"); quit()"
    EC=$?
    if [ $EC -ne 0 ]; then
      echo "srun_Fieldtrip_srcrec.sh: matlab exit code $EC, exiting"
      exit $EC
    fi
  fi

else
  # TODO interpret rawcur as comma-separated list of raws
  IFS_backup=$IFS
  IFS=','
  read -ra raws_tosrcrec <<< $rawcur
  IFS=$IFS_backup

  nraws=${#raws_tosrcrec[*]}
  rns_str='rawnames=['
  for (( i=0; i<=$(( $nraws -1 )); i++ )); do
    t=${raws_tosrcrec[$i]}
    rns_str=$rns_str'"'$t'"'
    if [ $i -lt $nraws-1 ]; then
      rns_str=$rns_str,
    fi
    #python3 $OSCBAGDIS_DATAPROC_CODE/run/run_collect_artifacts.py -r $t
    python $CODERUN/run_collect_artifacts.py -r $t --min_dur $MIN_DURATION_QUIET_ALLOWED $s


    EC=$?
    if [ $EC -ne 0 ]; then
      echo "srun_Fieldtrip_srcrec.sh: run_collect_artifacts exit code $EC, exiting"
      exit $EC
    fi
  done
  rns_str=$rns_str'];'

  vardefstr=$MATLAB_SCRIPT_PARAMS$rns_str
  echo $vardefstr
  #matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit"
  if [ $DO_SOURCE_RECONSTRUCTION -ne 0 ]; then
    matlab -nodisplay -nosplash -r "$vardefstr run(\"$OSCBAGDIS_DATAPROC_CODE/MEGsource_reconstruct_multiraw\"); quit()"
    EC=$?
    if [ $EC -ne 0 ]; then
      echo "srun_Fieldtrip_srcrec.sh: matlab exit code $EC, exiting"
      exit $EC
    fi
  fi
fi


#matlab -nodisplay -nosplash -r "run_ftsrc; quit"

#matlab -nodisplay -nosplash -r "try; $vardefstr; run_ftsrc.m; catch ME; fprintf(ME.identifier); end; quit"


# this would prevent the subsequent use of stuff in matlab
#matlab -r "prog arg1 arg2"
#
#which is equivalent to calling
#
#prog(arg1,arg2)
#
#from inside Matlab.

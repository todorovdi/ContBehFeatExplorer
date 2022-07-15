#!/bin/bash

# in bash Inside single quotes everything is preserved literally, without exception.
#
# in matlab single quotes define a character vector with size 1xN, where N is
# the number of characters between the quotes. double quotes define a string
# array with size 1x1 (i.e. scalar, regardless of how many characters it has).
# You can think of string arrays as a kind of container for character vectors.

#raws_comp
raw=$1

MERGE_RAWS=0      # actually not implemented yet when = 1, I need to merge in Fieldtrip and then de-merge for saving
INPUT_SUBDIR=""
OUTPUT_SUBDIR=""

DO_SOURCE_RECONSTRUCTION=1

ROI_TYPES='"'parcel_aal_surf'"'
#ROI_TYPES='"'HirschPt2011,2013'","'Thalamus'"'
#ROI_TYPES='"'parcel_aal_surf'","'Thalamus_L'","'Thalamus_R'"' 
#ROI_TYPES='"'HirschPt2011,2013'"'
#ROI_TYPES='"'Thalamus_L'","'Thalamus_R'"'

if [ -z ${vardefstr+x} ]; then
  vardefstr=""
fi
if [ -z ${MATLAB_SCRIPT_PARAMS+x} ]; then
  MATLAB_SCRIPT_PARAMS=""
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'input_rawname_type = ["resample", "notch", "highpass"];'
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'roi = {'$ROI_TYPES'};'
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS"use_data_afterICA=0;"
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'input_subdir="'$INPUT_SUBDIR'";'
  MATLAB_SCRIPT_PARAMS=$MATLAB_SCRIPT_PARAMS'output_subdir="'$OUTPUT_SUBDIR'";'
fi

if [ $MERGE_RAWS -eq 0 ]; then
  t=$raw
  python3 run_collect_artifacts.py -r $t

  vardefstr=$MATLAB_SCRIPT_PARAMS'rawnames=["'$t'"];'
  echo $vardefstr
  #exit 0
  #matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit"
  if [ $DO_SOURCE_RECONSTRUCTION -ne 0 ]; then
    matlab -nodisplay -nosplash -r "$vardefstr MEGsource_reconstruct_multiraw; quit()"
  fi
else
  # TODO interpret raw as comma-separated list of raws
  nraws=${#raws[*]}
  rns_str='rawnames=['
  for (( i=0; i<=$(( $nraws -1 )); i++ )); do
    t=${raws[$i]}
    rns_str=$rns_str'"'$t'"'
    if [ $i -lt $nraws-1 ]; then
      rns_str=$rns_str,
    fi
    python3 run_collect_artifacts.py -r $t
  done
  rns_str=$rns_str'];'

  vardefstr=$MATLAB_SCRIPT_PARAMS$rns_str
  echo $vardefstr
  #matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit"
  if [ $DO_SOURCE_RECONSTRUCTION -ne 0 ]; then
    matlab -nodisplay -nosplash -r "$vardefstr MEGsource_reconstruct_multiraw; quit()"
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

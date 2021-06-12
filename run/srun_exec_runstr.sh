#!/bin/bash
RUNSTRINGS_FN="_runstrings.txt"
RUNSTRING_IND=$1
mapfile -t RUNSTRINGS < $RUNSTRINGS_FN
RUNSTRING_CUR=${RUNSTRINGS[$RUNSTRING_IND]} 
R=$(eval echo $RUNSTRING_CUR)
ipython3 $R
#python3 $R

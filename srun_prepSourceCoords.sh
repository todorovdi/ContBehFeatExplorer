#!/bin/bash

#SRCS_TYPE=aal
SRCS_TYPE=HirschPt

echo "SRCS_TYPE=$SRCS_TYPE"

subjstrs=("S06" "S07" "S08" "S09" "S10")
subjstrs=("S08")
subjstrs=("S02" "S03")
#subjstrs=("S03")
subjstrs=("S01" "S02" "S03" "S04" "S05" "S06" "S07" "S08" "S09" "S10")
#subjstrs=("S01")
for t in ${subjstrs[@]}; do
  vardefstr="subjstr='$t'; skipPlot=1;"
  vardefstr="subjstr='$t'"
  if [ "$SRCS_TYPE" = "HirschPt" ]; then
    matlab -nodisplay -nosplash -r "$vardefstr; calcSrcCoords; quit"
  else
    matlab -nodisplay -nosplash -r "$vardefstr; parcel_script_aal; quit"
  fi

  #matlab -nodisplay -r "$vardefstr; calcSrcCoords; quit"
  #matlab -r "$vardefstr; calcSrcCoords; quit"
done

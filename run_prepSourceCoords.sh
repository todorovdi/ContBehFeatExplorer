#!/bin/bash
#subjstrs='["S06"]'
subjstrs='["S06"]'

subjstrs=("S06" "S07" "S08" "S09" "S10")
subjstrs=("S08")
subjstrs=("S02" "S03")
#subjstrs=("S03")
for t in ${subjstrs[@]}; do
  vardefstr="subjstr='$t'; skipPlot=1;"
  matlab -nodisplay -nosplash -r "$vardefstr; calcSrcCoords; quit"
done

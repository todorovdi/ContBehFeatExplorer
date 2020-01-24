#!/bin/bash
#subjstrs='["S06"]'
subjstrs='["S06"]'

subjstrs=("S06", "S07", "S08", "S09", "S10")
subjstrs=('"S08"')
for t in ${subjstrs[@]}; do
  vardefstr="subjstr=$t; skipPlot=1;"
  matlab -nodisplay -nosplash -r "$vardefstr; ftplottest; quit"
done

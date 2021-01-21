#!/bin/bash


raws_off=(S01_off_hold S02_off_hold S03_off_hold S04_off_hold S05_off_hold S07_off_hold )  
raws_off_compl=(S01_off_move S02_off_move S03_off_move S04_off_move S05_off_move S07_off_move )  
raws_on=(S01_on_hold S02_on_hold S04_on_hold S05_on_hold S07_on_hold )  
raws_on_compl=(S01_on_move S02_on_move S04_on_move S05_on_move S07_on_move )  

#subjstr = 'S01';
#typestr = 'move';
#medstr  = 'off';
#subjstrs='["S01", "S02", "S03", "S04"]'
#subjstrs='["S04", "S05", "S06"]'
#subjstrs='["S05", "S06", "S07", "S08", "S09", "S10"]'
#subjstrs='["S02"]'
#subjstrs='["S03"]'
#subjstrs='["S08","S09"]'
#subjstrs='["S01"]'
subjstrs='["S06"]'
subjstrs='["S05", "S06", "S07","S08","S09","S10"]'
subjstrs='["S01", "S02", "S03"]'
subjstrs='["S02", "S03"]'
subjstrs='["S01"]'
subjstrs='["S01", "S02", "S03", "S04", "S05", "S07"]'
subjstrs='["S04", "S05", "S07"]'
subjstrs='["S01", "S02", "S03"]'
subjstrs='["S05"]'
subjstrs='["S04"]'
subjstrs='["S02", "S03", "S04", "S05", "S07" , "S10" ]'
subjstrs='[ "S06" , "S08", "S09" ]'
subjstrs='["S01", "S02", "S03"]'
subjstrs='["S01", "S02", "S03", "S04", "S05", "S07", "S09", "S10"]'
subjstrs='[ "S06" , "S08" ]'
subjstrs='["S01"]'
# in bash Inside single quotes everything is preserved literally, without exception.
#
# in matlab single quotes define a character vector with size 1xN, where N is
# the number of characters between the quotes. double quotes define a string
# array with size 1x1 (i.e. scalar, regardless of how many characters it has).
# You can think of string arrays as a kind of container for character vectors.
subjs=(S01 S02 S04 S05 S07)
subj="S99"
subjstrs='["S99"]'
vardefstr="subjstrs=$subjstrs"
#vardefstr="subjstrs=$subjstrs;"' medconds=["off"];'' tasks=["hold"];'
#vardefstr=""
#echo "$vardefstr"


      raws=(S01_off_hold S01_on_hold S02_off_hold S02_on_hold S04_off_hold S04_on_hold S05_off_hold S05_on_hold S07_off_hold S07_on_hold)
raws_compl=(S01_off_move S01_on_move S02_off_move S02_on_move S04_off_move S04_on_move S05_off_move S05_on_move S07_off_move S07_on_move)
#raws=      (S01_off_hold S01_on_hold)
#raws_compl=(S01_off_move S01_on_move)

      raws=(S01_off_hold)
raws_compl=(S02_off_hold)

#      raws=(S01_off_hold)
#raws_compl=(S01_off_move)
#
#      raws=(S01_on_hold S02_off_hold S02_on_hold)
#raws_compl=(S01_on_move S02_off_move S02_on_move)


nraws=${#raws[*]}

#matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit"

for (( i=0; i<=$(( $nraws -1 )); i++ )); do
  t=${raws[$i]}
  t_compl=${raws_compl[$i]}

  python3 run_collect_artifacts.py -r $t
  python3 run_collect_artifacts.py -r $t_compl

  vardefstr='rawnames=["'$t'","'$t_compl'"]'
  #matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit"
  matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit()"
done

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

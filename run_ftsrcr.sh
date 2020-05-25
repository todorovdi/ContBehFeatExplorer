#!/bin/bash
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
#subjstrs='["S01"]'
vardefstr="subjstrs=$subjstrs"
#vardefstr=""
echo "$vardefstr"

matlab -nodisplay -nosplash -r "$vardefstr; MEGsource_reconstruct_multiraw; quit"
#matlab -nodisplay -nosplash -r "run_ftsrc; quit"

#matlab -nodisplay -nosplash -r "try; $vardefstr; run_ftsrc.m; catch ME; fprintf(ME.identifier); end; quit"

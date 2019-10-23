#!/bin/bash
#subjstr = 'S01';
#typestr = 'move';
#medstr  = 'off';
subjstrs='["S01", "S02", "S03"]'
subjstrs='["S02"]'
subjstrs='["S03"]'
#subjstrs='["S01"]'
vardefstr="subjstrs=$subjstrs"
echo "$vardefstr"

matlab -nodisplay -nosplash -r "$vardefstr; run_ftsrc; quit"
#matlab -nodisplay -nosplash -r "run_ftsrc; quit"

#matlab -nodisplay -nosplash -r "try; $vardefstr; run_ftsrc.m; catch ME; fprintf(ME.identifier); end; quit"

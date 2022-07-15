#!/bin/bash
# should be called like list_concise 10 '*best*'
# yes, with single quotes encosing wildcards
if [[ $# -ne 2 ]]; then 
  echo need two arguments = numlines and search pattern
  exit 1
fi
numlines=$1
search_pattern=$2
echo "$1, $2"
#$sort="sort -r -n -k 3"
srt="sort -n -k 3"
#stat -c '%.19y %n %Y' * | sort -n -k 3  | tail -$numlines | awk {'print $1"\t" $2"\t" $3'}
numlines=`stat -c '%.19y %n %Y %s %F' $search_pattern | grep regular | wc -l`
echo "NUMLINES=$numlines"
stat -c '%.19y %n %Y %s %F' $search_pattern | grep regular | $srt  | tail -$numlines | awk {'print $1"\t" $2"\t" ($5/1048576)"mb\t" $3'} 
echo "  dirs:"
stat -c '%.19y %n %Y %F' $search_pattern | grep directory | $srt  | tail -$numlines | awk {'print $1"\t" $2"\t" $3'} 
#echo "echoing $search_pattern"
#stat -c "%.19y %n %Y" $search_pattern 
#output: 2022-04-14 12:15:31 best_LFP_info_both_sides.json 1649931331 89497

# for dir size du -sh dirnam

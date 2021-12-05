numlines=$1
search_pattern=$2
#TODO: chack parameters
#TODO: print filesize somehow
#stat -c '%.19y %n %Y' * | sort -n -k 3  | tail -$numlines | awk {'print $1"\t" $2"\t" $3'}
stat -c '%.19y %n %Y' $search_pattern | sort -n -k 3  | tail -$numlines | awk {'print $1"\t" $2"\t" $3'}
#echo "echoing $search_pattern"
#stat -c "%.19y %n %Y" $search_pattern 

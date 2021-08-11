numlines=$1
search_pattern=$2
#stat -c '%.19y %n %Y' * | sort -n -k 3  | tail -$numlines | awk {'print $1"\t" $2"\t" $3'}
stat -c '%.19y %n %Y' $search_pattern | sort -n -k 3  | tail -$numlines | awk {'print $1"\t" $2"\t" $3'}

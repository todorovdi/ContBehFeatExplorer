#
slurmout_path=$CODE/slurmout
narg=$#
echo narg=$narg
if [ $narg -le 1 ]; then
  if [ $narg -eq 0 ]; then
    numlines=15
  else 
    numlines=$1
  fi

  lastfilestr=`ls -ltr "$slurmout_path" | grep -v ^d | tail -1`
  echo "$lastfilestr"
  lastfile=`echo $lastfilestr | awk '{ print $NF }'`
  export lastfile=$slurmout_path/"$lastfile"
else
  ind=$1
  numlines=$2
  lastfilestr=`ls -ltr "$slurmout_path"/*_$ind.out | grep -v ^d | tail -1`
  lastfile=`echo $lastfilestr | awk '{ print $NF }'`
  export lastfile=$slurmout_path/"$lastfile"
fi

#echo "$lastfilestr"
#lastfile=`echo "$lastfilestr" | tail -1`


#lastfile=`find slurmout -maxdepth 1 -not -type d | tail -1`
echo ":::Showing end of lastfile=$lastfile"
echo ""
tail -$numlines ${lastfile[0]} 

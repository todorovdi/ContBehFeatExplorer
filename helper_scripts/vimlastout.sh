lastfilestr=`ls -ltr slurmout | grep -v ^d | tail -1`

#echo "$lastfilestr"
#lastfile=`echo "$lastfilestr" | tail -1`

lastfile=`echo $lastfilestr | awk '{ print $NF }'`
lastfile=slurmout/"$lastfile"

#lastfile=`find slurmout -maxdepth 1 -not -type d | tail -1`
echo ":::Showing end of lastfile=$lastfile"
echo ""
vim -R ${lastfile[0]} 

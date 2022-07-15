#!/usr/bin/bash
#!! it does not check if the job with given index exist in the last job submitted
narg=$#
if [ $narg -ge 1 ]; then
  ind=$1                
else
  ind=""
fi

# cancels last job found in squeue (entire job, not just one index)
job_and_ind_all=`squeue -u todorov1 -r | grep ' R\|PD\|CG ' | awk '{print $1}'`
job_and_ind=`squeue -u todorov1 -r | grep ' R\|PD ' | head -1 | awk '{print $1}'`
arr=(${job_and_ind//_/ })
job=${arr[0]}
ind_=${arr[1]}

job_and_ind_all=(${job_and_ind_all// / })

#echo ${job_and_ind_all[*]}

#if [[ " ${array[*]} " =~ " ${value} " ]]; then
#    # whatever you want to do when array contains value
#fi
#
#if [[ ! " ${array[*]} " =~ " ${value} " ]]; then
#    # whatever you want to do when array doesn't contain value
#fi

if [ $narg -ge 1 ]; then
  #tj="$job"_"$ind"; echo $tj
  last_job_and_ind=""
  for job_and_ind_cur in "${job_and_ind_all[@]}"
  do
    subarr=(${job_and_ind_cur//_/ })
    job_cur=${subarr[0]}
    ind_cur=${subarr[1]}
    #echo $job_and_ind_cur "ff"
    #if [[ "$job_and_ind_cur" == "$tj" ]] ; then
    if [[ "$ind_cur" == "$ind" ]] ; then
      #echo "LAAL"
      last_job_and_ind=$job_and_ind_cur
    fi
  done

  echo last_job_and_ind = $last_job_and_ind
  if [ "$last_job_and_ind" = "" ]; then
      echo "Empty last_job_and_ind, exiting"
      exit 1
  fi

  #if [[ ! " ${job_and_ind_all[*]} " =~ " ${tj} " ]]; then
  #    # whatever you want to do when array doesn't contain value
  #    echo "Last job is $job but it does not have index $ind"
  #    exit 1
  #fi
fi


if [ $narg -ge 1 ]; then
  ind_=""
  echo "Planning cancelling $last_job_and_ind (index unused=$ind_, used=$ind)"
  scancel $last_job_and_ind; echo "Cancelled $last_job_and_ind"
else
  echo "Planning cancelling $job (index unused=$ind_)"
  scancel $job; echo "Cancelled $job"
fi

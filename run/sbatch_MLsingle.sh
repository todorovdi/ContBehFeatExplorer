#!/bin/bash
# budget account where contingent is taken from
###nodes=2, ntasksper=256, cpu=1 does not work better than nodes = 1 with other fixed

##### run_genfeats needs about 1.5h

#SBATCH --account=icei-hbp-2020-0012
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=batch

##SBATCH --time=10:00:00
##SBATCH --mem=128G

#SBATCH --time=06:30:00
#SBATCH --mem=50G
#SBATCH --array=0-88

##SBATCH --mem=80G
##SBATCH --partition=batch
##SBATCH --time=4:00:00
##SBATCH --mem=80G

## max array size = 256  (as shown by scontrol show config)
## 22 decent dataset 
## around sixty jobs at once
## sacctmgr list associations
## sacctmgr show qos

#SBATCH --output ../slurmout/ML_%A_%a.out
#SBATCH --error ../slurmout/ML_%A_%a.out
# if keyword omitted: Default is slurm-%j.out in
# the submission directory (%j is replaced by
# the job ID).

#SBATCH --mail-type=ALL
#SBATCH --mail-user=todorovdi@gmail.com

##SBATCH --array=0,3
##SBATCH --array=166  modLFP for S03

# *** start of job script ***
##source set_oscabagdis_env_vars.sh

##RUNSTRINGS_FN="_runstrings.txt"
##mapfile -t RUNSTRINGS < $RUNSTRINGS_FN
##num_runstrings=${#RUNSTRINGS[*]}

## export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
### srun is for MPI programs
###  sbatch <jobscript> 
## srun <executable>

#srun --exclusive -n 128 ./mpi-prog1 &
#srun --exclusive -n 128 ./mpi-prog2 &
#wait

jutil env activate -p icei-hbp-2020-0012

JOBID=$SLURM_JOB_ID
ID=$SLURM_ARRAY_TASK_ID


##export PROJECT=$HOME/shared/OSCBAGDIS/data_proc
##export DATA=$HOME/shared/
export PROJECT_DIR=$PROJECT/OSCBAGDIS
export CODE=$PROJECT_DIR/data_proc_code
export DATA_DUSS=$PROJECT_DIR/data_duss
export OUTPUT_OSCBAGDIS=$PROJECT_DIR/output
export OSCBAGDIS_DATAPROC_CODE=$PROJECT_DIR/data_proc_code

export PATH=$PATH:$HOME/.local/bin
##export PYTHONPATH=$OSCBAGDIS_DATAPROC_CODE
#
#module purge
#module load scikit
#module load Python-Neuroimaging
#module load SciPy-Stack
module load Stages/2022
module load GCC
module load R
module load Python

echo "DATA_DUSS=$DATA_DUSS"

source $CODE/__workstart.sh
pwd

export PYTHONPATH=$PYTHONPATH:$PROJECT/OSCBAGDIS/LOCAL/lib/python3.9/site-packages

EXIT_IF_ANY_FAILS=0
NFAILS=0
NRUNS=0

RUNSTRINGS_FN="$CODE/run/_runstrings_ML.txt"

SHIFT_ID=0
#EFF_ID=$((ID+SHIFT_ID))
#echo "Running job array number: ${ID} (effctive_id = $EFF_ID) on $HOSTNAME,  $SLURM_JOB_ID, $SLURM_ARRAY_JOB_ID"
#$OSCBAGDIS_DATAPROC_CODE/run/srun_exec_runstr.sh $RUNSTRINGS_FN $SLURM_ARRAY_JOB_ID $EFF_ID

MAXJOBS=256 # better this than 64, otherwise more difficult on the level of indtool

NUMRS=`wc -l $RUNSTRINGS_FN | awk '{print $1;}'`
echo "Start now"
echo "SBATCH TYPE: CPU SINGLERUN"
while [ $NUMRS -gt $SHIFT_ID ]; do
  EFF_ID=$((ID+SHIFT_ID))
  if [ $EFF_ID -ge $NUMRS ]; then
    break
  fi
  echo "Running job array number: ${ID} (effctive_id = $EFF_ID) on $HOSTNAME,  $SLURM_JOB_ID, $SLURM_ARRAY_JOB_ID"
  #python -c "1/0"

  $OSCBAGDIS_DATAPROC_CODE/run/srun_exec_runstr.sh $RUNSTRINGS_FN $SLURM_ARRAY_JOB_ID $EFF_ID $ID
  EXCODE=$?

  #EXCODE=0
  echo "---!!!--- Current run error code: $EXCODE"
  if [[ $EXCODE -ne 0 ]]; then 
    NFAILS=$((NFAILS + 1))
    echo "NFAILS=$NFAILS"
  fi

  if [[ $EXCODE -ne 0 ]] && [[ $EFF_ID -eq 0 ]]; then
    echo "Exiting due to bad error code in test :("
    exit $EXCODE
  fi

  if [[ $EXCODE -ne 0 ]] && [[ $EXIT_IF_ANY_FAILS -ne 0 ]]; then
    echo "Exiting due to bad error code :("
    exit $EXCODE
  fi
  SHIFT_ID=$((SHIFT_ID + MAXJOBS))

  echo "----------------"
  echo "----------------"
  echo "----------------"
  echo "----------------"
  echo "FINISHED job array number: ${ID} (effctive_id = $EFF_ID) on $HOSTNAME,  $SLURM_JOB_ID, $SLURM_ARRAY_JOB_ID"
  NRUNS=$((NRUNS + 1))
  ##########################  ONLY FOR RUNNING OF INDIVID JOBS
  break # BECAUSE WE ARE IN single .sh file
  ##########################
done


echo "---!!!--- END OF EVERYTHING: failed $NFAILS of $NRUNS "
echo "---!!!--- End error code: $NFAILS"
exit $NFAILS  # added afterrunning job 233971

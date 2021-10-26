#!/bin/bash
# budget account where contingent is taken from
###nodes=2, ntasksper=256, cpu=1 does not work better than nodes = 1 with other fixed

##### run_genfeats needs about 1.5h

#SBATCH --account=icei-hbp-2020-0012
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

#SBATCH --time=23:00:00
#SBATCH --partition=batch
#SBATCH --mem=80G

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
#SBATCH --array=0-42

# *** start of job script ***
##source set_oscabagdis_env_vars.sh

##RUNSTRINGS_FN="_runstrings.txt"
##mapfile -t RUNSTRINGS < $RUNSTRINGS_FN
##num_runstrings=${#RUNSTRINGS[*]}

## export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
### srun is for MPI programs
###  sbatch <jobscript> 
## srun <executable>

jutil env activate -p icei-hbp-2020-0012

JOBID=$SLURM_JOB_ID
ID=$SLURM_ARRAY_TASK_ID
SHIFT_ID=0
EFF_ID=$((ID+SHIFT_ID))
echo "Running job array number: ${ID} (effctive_id = $EFF_ID) on $HOSTNAME,  $SLURM_JOB_ID, $SLURM_ARRAY_JOB_ID"


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
module load R

echo "DATA_DUSS=$DATA_DUSS"

source $CODE/__workstart.sh
pwd

RUNSTRINGS_FN="$CODE/run/_runstrings_ML.txt"
$OSCBAGDIS_DATAPROC_CODE/run/srun_exec_runstr.sh $RUNSTRINGS_FN $SLURM_ARRAY_JOB_ID $EFF_ID

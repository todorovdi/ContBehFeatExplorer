#!/bin/bash
# budget account where contingent is taken from
###nodes=2, ntasksper=256, cpu=1 does not work better than nodes = 1 with other fixed

##### run_genfeats needs about 1.5h

#SBATCH --account=icei-hbp-2020-0012
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

###SBATCH --time=23:00:00
## usually finished in around 30 min
#SBATCH --time=03:00:00
##SBATCH --partition=gpus
#SBATCH --partition=batch
#SBATCH --mem=64G

##SBATCH --partition=batch
##SBATCH --mem=60G

## max array size = 256  (as shown by scontrol show config)
## 22 decent dataset 
## around sixty jobs at once
## sacctmgr list associations
## sacctmgr show qos

#SBATCH --output /p/project/icei-hbp-2020-0012/slurmout/ML_%A_%a.out
#SBATCH --error /p/project/icei-hbp-2020-0012/slurmout/ML_%A_%a.out
# if keyword omitted: Default is slurm-%j.out in
# the submission directory (%j is replaced by
# the job ID).

#SBATCH --mail-type=ALL
#SBATCH --mail-user=todorovdi@gmail.com

#SBATCH --array=0-21

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
export MNE_USE_CUDA=1


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
module load CUDA

echo "DATA_DUSS=$DATA_DUSS"

source $CODE/__workstart.sh
pwd

export PYTHONPATH=$PYTHONPATH:$PROJECT/OSCBAGDIS/LOCAL/lib/python3.9/site-packages

echo "Shell is $SHELL"

RUNSTRINGS_FN="$CODE/run/_runstrings_resave.txt"
# this one runs the runstring directly, not runs python with some argument
$OSCBAGDIS_DATAPROC_CODE/run/srun_exec_runstr0.sh $RUNSTRINGS_FN $SLURM_ARRAY_JOB_ID $EFF_ID $ID

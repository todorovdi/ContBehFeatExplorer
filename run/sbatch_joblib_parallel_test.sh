#!/bin/bash
# budget account where contingent is taken from
#SBATCH --account=icei-hbp-2020-0012
#
#SBATCH --nodes=5
# can be omitted if --nodes and --ntasks-per-node
##SBATCH --ntasks=<no of tasks (MPI processes)>

# if keyword omitted: Max. 256 tasks per node
#SBATCH --ntasks-per-node=3

# for OpenMP/hybrid jobs only
#SBATCH --cpus-per-task=7

##### run_genfeats needs about 1.5h

#SBATCH --time=0:20:00
#SBATCH --partition=batch
#SBATCH --mem=5G

##SBATCH --array=0-10

#SBATCH --output slurmout/test_%A_%a.out
#SBATCH --error slurmout/test_%A_%a.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=todorovdi@gmail.com

#SBATCH -J ipy_engines      #job name
###SLURM_JOB_ID=12345

### len(client)=15, joblib.cpu_count()=256, mpr.cpu_count()=256
profile=job_${SLURM_JOB_ID}

echo "Creating profile_${profile}"
ipython profile create ${profile}

ipcontroller --ip="*" --profile=${profile} &
sleep 10

#srun: runs ipengine on each available core
srun ipengine --profile=${profile} --location=$(hostname) &
sleep 25

SCRIPT_NAME=$1
SCRIPT_NAME="run_joblib_parallel_test.py"

echo "Launching job for script $SCRIPT_NAME"
python3 $SCRIPT_NAME -p ${profile}

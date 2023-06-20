#!/bin/bash

# Job name
#SBATCH --job-name=ldau
# Number of Nodes
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=results/aflow_fix/ef/no_ldau_kfold/%A-%a.out
# Number of array jobs, and maximum number of simultaneous jobs (%n)
#SBATCH --array=0-9%10
# Number of processes per Node
#SBATCH --ntasks-per-node=1
# Number of CPU-cores per task
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=<bechtelt@physik.hu-berlin.de>

#SBATCH --time=0-23:59:59 # days-hh:mm:ss.

module load cuda
python3 scripts/crossval/train_kfold.py \
--config=jraph_MPEU_configs/aflow_ef_clean.py \
--workdir=results/aflow_fix/ef/no_ldau_kfold/id${SLURM_ARRAY_TASK_ID} \
--i_fold=${SLURM_ARRAY_TASK_ID} \
--n_splits=10

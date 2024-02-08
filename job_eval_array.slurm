#!/bin/bash

# Job name
#SBATCH --job-name=eval
# Number of Nodes
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=results/aflow/kfold_Ef/%A-%a.out
# Number of array jobs, and maximum number of simultaneous jobs (%n)
#SBATCH --array=0-10%10
# Number of processes per Node
#SBATCH --ntasks-per-node=1
# Number of CPU-cores per task
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=<bechtelt@physik.hu-berlin.de>

#SBATCH --time=0-23:59:59 # days-hh:mm:ss.

module load cuda
python3 scripts/plotting/error_analysis.py \
--file=results/aflow/kfold_Ef/id${SLURM_ARRAY_TASK_ID} \

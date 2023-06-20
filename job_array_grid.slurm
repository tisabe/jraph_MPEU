#!/bin/bash

# Job name
#SBATCH --job-name=grid_dropout
# Number of Nodes
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=results/aflow_fix/ef/dropout_grid/%A-%a.out
# Number of array jobs, and maximum number of simultaneous jobs (%n)
#SBATCH --array=0-47%20
# Number of processes per Node
#SBATCH --ntasks-per-node=1
# Number of CPU-cores per task
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=<bechtelt@physik.hu-berlin.de>

#SBATCH --time=0-23:59:59 # days-hh:mm:ss.

module load cuda
python3 scripts/crossval/crossval_grid.py \
--config=jraph_MPEU_configs/aflow_dropout_grid.py \
--workdir=results/aflow_fix/ef/dropout_grid/id${SLURM_ARRAY_TASK_ID} \
--index=${SLURM_ARRAY_TASK_ID}

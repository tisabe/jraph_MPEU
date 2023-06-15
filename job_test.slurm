#!/bin/bash

# Job name
#SBATCH --job-name=egap_pbj
# Number of Nodes
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=results/aflow_fix/egap_pbj/%A.out
# Number of processes per Node
#SBATCH --ntasks-per-node=1
# Number of CPU-cores per task
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=<bechtelt@physik.hu-berlin.de>

#SBATCH --time=1-00:00:00 # days-hh:mm:ss.

module load cuda
python3 scripts/main.py \
--config=jraph_MPEU_configs/aflow_egap_pbj.py \
--workdir=results/aflow_fix/egap_pbj

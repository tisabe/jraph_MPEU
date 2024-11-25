#!/bin/bash

# Job name
#SBATCH --job-name=qm9_lowlr
# Number of Nodes
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=results/qm9/dropout_lowlr/%A.out
# Number of processes per Node
#SBATCH --ntasks-per-node=1
# Number of CPU-cores per task
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=<bechtelt@physik.hu-berlin.de>

#SBATCH --time=1-00:00:00 # days-hh:mm:ss.

module load cuda
python3 scripts/train.py --config=jraph_MPEU_configs/default.py \
--workdir=results/qm9/dropout_lowlr \
--config.dropout_rate=0.1 \
--config.init_lr=2e-5

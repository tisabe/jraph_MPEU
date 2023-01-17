#!/bin/bash -l
# set the working directory as a variable:
# workdir='./results/aflow/test/run2'
# Standard output and error:
#SBATCH -o ./output_slurm/job.out.%j
#SBATCH -e ./output_slurm/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J aflow
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-core=1
#SBATCH --mem=32000        # Request 32 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=23:00:00

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=10

python scripts/main.py --workdir=./results/aflow/classify_0 \
--config=jraph_MPEU_configs/aflow_classify.py

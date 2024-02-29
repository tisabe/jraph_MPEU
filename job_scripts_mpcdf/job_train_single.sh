#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./output_slurm/singlejob.%j.out
#SBATCH -e ./output_slurm/singlejob.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J qm9_painn
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-core=1
#SBATCH --mem=32000        # Request 32 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=12:00:00
#SBATCH --partition=gpudev

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python scripts/main.py \
--workdir=./results/qm9/U0/painn \
--config=jraph_MPEU_configs/qm9_painn.py 

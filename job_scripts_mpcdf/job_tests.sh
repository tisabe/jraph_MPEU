#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./output_slurm/test.%j.out
#SBATCH -e ./output_slurm/test.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J tests
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=1
#SBATCH --mem=8        # Request 32 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=00:10:00

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=10

python tests/test_models.py

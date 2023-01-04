#!/bin/bash -l
# set the working directory as a variable:
# workdir='./results/aflow/test/run2'
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J train_single
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-core=1
#SBATCH --mem=32000        # Request 32 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=00:10:00

module purge
module load anaconda/3/2020.02
module load intel/21.2.0 cuda/11.4 cudnn/8.2.4

export OMP_NUM_THREADS=10

python3.7 scripts/main.py --workdir=./results/aflow/test/run2 \
--config=jraph_MPEU_configs/default_aflow_test.py

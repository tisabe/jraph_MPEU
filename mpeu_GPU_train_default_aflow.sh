#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_gpu
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   #   providing GPU
#SBATCH --gres=gpu:a100:4    # Make sure you have a full node to yourself. Most probably using only one GPU.
#SBATCH --cpus-per-task=72
#SBATCH --mem=0        # Request 180 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=00:10:00
#SBATCH --reservation=handson

module purge
module load anaconda/3/2020.02
module load intel/21.2.0 cuda/11.4 cudnn/8.2.4
module load nsight_systems/2022

export OMP_NUM_THREADS=72

nsys profile python3.7 scripts/main.py --workdir=results/aflow/default_two_steps_10k --config=jraph_MPEU_configs/default_aflow_test.py


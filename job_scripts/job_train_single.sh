#!/bin/bash -l
# set the working directory as a variable:
# workdir='./results/aflow/test/run2'
# Standard output and error:
#SBATCH -o ./output_slurm/job.%j.out
#SBATCH -e ./output_slurm/job.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J classify
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-core=1
#SBATCH --mem=32000        # Request 32 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=00:30:00

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python scripts/crossval/crossval_mc.py \
--workdir=./results/aflow/egap_rand_search_test/id1 \
--config=jraph_MPEU_configs/aflow_rand_search_egap.py \
--index=1 \
--split_file=./results/aflow/classify_new_dropout/splits_ins.json

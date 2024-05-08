#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./output_slurm/singlejob.%j.out
#SBATCH -e ./output_slurm/singlejob.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J mpeu_mp
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

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python scripts/main.py \
--workdir=./results/aflow_x_mp/ef/train_combined \
--config=jraph_MPEU_configs/default_mp.py \
--config.label_str=ef_atom \
--config.data_file=databases/aflow_x_matproj/graphs_12knn_vec.db

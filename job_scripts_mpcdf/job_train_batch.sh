#!/bin/bash -l
# specify the indexes (max. 30000) of the job array elements (max. 300 - the default job submit limit per user)
#SBATCH --array=1-10        # indices are inclusive
# Standard output and error:
#SBATCH -o ./output_slurm/train_batch_%A_%a.out
#SBATCH -e ./output_slurm/train_batch_%A_%a.err 
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J pbj_batch
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=1
#SBATCH --mem=8000         # Requested main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=12:00:00     # 12h should be enough for any configuration

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python scripts/main.py \
--workdir=./results/qm9/U0/uq_ensemble/id${SLURM_ARRAY_TASK_ID} \
--config=jraph_MPEU_configs/default.py \
--config.seed_weights=${SLURM_ARRAY_TASK_ID} \
--config.shuffle_val_seed=${SLURM_ARRAY_TASK_ID} \
--config.model_str=MPEU_uq \

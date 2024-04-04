#!/bin/bash -l
# specify the indexes (max. 30000) of the job array elements (max. 300 - the default job submit limit per user)
#SBATCH --array=1-5       # indices are inclusive
# Standard output and error:
#SBATCH -o ./output_slurm/eval_%A_%a.out
#SBATCH -e ./output_slurm/eval_%A_%a.err 
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J eval_batch
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=1
#SBATCH --mem=4000         # Request 32 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=12:00:00     # 12h should be enough for any configuration

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python scripts/plotting/error_analysis.py \
--file=./results/qm9/U0/uq_ensemble/id${SLURM_ARRAY_TASK_ID} \
--plot=None

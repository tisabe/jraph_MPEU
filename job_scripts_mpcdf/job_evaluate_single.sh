#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./output_slurm/evaljob.%j.out
#SBATCH -e ./output_slurm/evaljob.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J eval_mp_co
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

srun python scripts/plotting/error_analysis.py \
--file=results/aflow_x_mp/ef/mp_infer_combined/ --label=ef --plot=nothing

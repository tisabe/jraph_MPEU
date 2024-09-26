#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./output_slurm/convertdb.%j.out
#SBATCH -e ./output_slurm/convertdb.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J data_convert
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-core=1
#SBATCH --mem=16000         # Request main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=4:00:00     # time limit in hours

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python scripts/data/combine_db.py

#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./output_slurm/datajob.out.%j
#SBATCH -e ./output_slurm/datajob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J data_convert
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-core=1
#SBATCH --mem=32000         # Request 32 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=1:00:00     # 1 hour

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python scripts/data/aflow_to_graphs.py \
-f aflow/egaps_eform_all.csv -o aflow/graphs_all_12knn.db -cutoff_type knearest \
-cutoff 12.0

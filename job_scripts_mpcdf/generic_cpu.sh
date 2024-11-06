#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./output_slurm/generic_cpu.%j.out
#SBATCH -e ./output_slurm/generic_cpu.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J generic_cpu
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=1
#SBATCH --mem=16000         # Request main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=24:00:00     # time limit in hours

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python scripts/plotting/combined_csv_analysis.py 
--paths=results/aflow/egap/painn/rand_search_best/result_3m.csv,results/aflow/ef/painn/rand_search_best/result_3m.csv,results/aflow/egap/mpeu/classify/result_3m.csv
--labels=egap,ef,egap_class
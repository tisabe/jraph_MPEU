#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./output_slurm/evaljob.%j.out
#SBATCH -e ./output_slurm/evaljob.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J eval
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-core=1
#SBATCH --mem=80000        # Request x MB of main memory.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=12:00:00

# load the environment with modules and python packages
cd ~/envs ; source ~/envs/activate_jax.sh
cd ~/jraph_MPEU

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python scripts/eval.py \
--workdir=results/aflow/egap/mpeu/classify \
--results_path=result_3m.csv \
--data_path=databases/aflow/eform_all_graphs_202409.db \
--mc_dropout=False \
--ensemble=False \

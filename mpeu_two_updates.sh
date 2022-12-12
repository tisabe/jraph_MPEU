#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_cpu_1m_steps
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --cpus-per-task=72
#SBATCH --mem=0        # Specify 0 memory to make node is for ourself.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=24:00:00


module purge
module load anaconda/3/2020.02
module load intel/21.2.0 cuda/11.4 cudnn/8.2.4
module load nsight_systems/2022

export OMP_NUM_THREADS=72

# export JAX_LOG_COMPILES=1
python3.7 scripts/main.py --workdir=results/aflow/default_aflow_fe_cpu_1M_steps_profiling_v2 --config=jraph_MPEU_configs/default_aflow_profile_test.py

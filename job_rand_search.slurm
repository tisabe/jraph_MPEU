#!/bin/bash -l
# set the working directory as a variable:
# workdir='./results/aflow/test/run3'
# mkdir $workdir
# mkdir ./results/aflow/test/run5
# specify the indexes (max. 30000) of the job array elements (max. 300 - the default job submit limit per user)
#SBATCH --array=0-99%50
# Standard output and error:
#SBATCH -o ./output_slurm/job.out.%A_%a        # Standard output, %A = job ID, %a = job array index
#SBATCH -e ./output_slurm/job.err.%A_%a        # Standard error, %A = job ID, %a = job array index
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_mc
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8000        # Request 8 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=23:00:00

module purge
module load anaconda/3/2020.02
module load intel/21.2.0 cuda/11.4 cudnn/8.2.4

export OMP_NUM_THREADS=10

python3.7 scripts/crossval/crossval_mc.py \
--workdir=./results/aflow/ef/rand_search/id${SLURM_ARRAY_TASK_ID} \
--config=jraph_MPEU_configs/aflow_rand_search_ef.py \
--index=${SLURM_ARRAY_TASK_ID} \

#!/bin/bash

# Job name
#SBATCH --job-name=aflow_data
# Number of Nodes
#SBATCH --nodes=1
# Number of processes per Node
#SBATCH --ntasks-per-node=1
# Number of CPU-cores per task
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=<bechtelt@physik.hu-berlin.de>

#SBATCH --time=0-12:00:00 # days-hh:mm:ss.

python3 scripts/data/asedb_to_graphs.py -f matproj/mp2018.db -o matproj/mp2018_graphs.db -cutoff_type knearest -cutoff 24.0
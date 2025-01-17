# jraph_MPEU

Message Passing Graph Neural Network with Edge Updates in Jraph.

This code implements tools for running and analysing a neural architecture search (NAS), focused on the regression of electronic band gaps and formation energies of solids in the AFLOW materials database. There are also some scripts to work with Materials Project data (the often used MP2018 snapshot), and the benchmark QM9 data.

Main architectures implemented here are:

"Neural Message Passing with Edge Updates for Predicting Properties of Molecules and Materials" https://arxiv.org/pdf/1806.03146.pdf
and
"PaiNN" https://proceedings.mlr.press/v139/schutt21a.html (Thanks to Gianluca Galletti for the porting into JAX. Check out https://github.com/gerkone/painn-jax/.)

All this code is experimental, run at your own risk!

## How to get this running:

### Python installation and libraries

1. We recommend using a Conda environment to create a Python 3.11 installation, from which a virtual environment (venv) is created:
    1.1 Install Conda as described in https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
    1.2 Create a Conda environment:
    `>>> conda create --name py3-11 python=3.11`
    1.3 Activate the environment:
    `>>> conda activate py3-11`
    1.4 create and activate a python venv:
    `>>> python -m venv venv`
    `>>> conda deactivate`
    `>>> source venv/bin/activate`
    1.5 update pip:
    `>>> pip install --upgrade pip`
2. Install the right JAX version for you, as described in https://jax.readthedocs.io/en/latest/installation.html (if you are in the venv, you can leave out the `-U` option)
    - The cpu version works with this code, but is probably too slow for most applications, so gpu version of JAX is recommended
3. Install the library in this repository:
    `>>> pip install -e .`
4. Install the rest of the required libraries:
    `>>> pip install -r requirements.txt`

### AFLOW dataset

1. pull the "benchmark" AFLOW data using:
`>>> python scripts/data/get_aflow_csv.py`
2. convert the csv data into a ASE database file (also pre-computes graph data, this might take some minutes):
`>>> python scripts/data/aflow_to_graphs.py`
Here there are some options for file names and how the graph adjacency should be generated.

### QM9 dataset

This will be added in a future update of this README.

### Training

`>>> python scripts/train.py --workdir=results/my_first_run --config=jraph_MPEU_configs/aflow_ef_default.py`



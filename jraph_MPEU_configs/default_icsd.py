"""Define the default hyperparameters for model and training."""

import ml_collections
from jraph_MPEU_configs.default_aflow import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = get_config_super() # inherit from default mp config

    config.data_file = 'aflow/icsd_energies_graphs_6A.db'
    #config.selection = 'natoms<80'
    config.label_str = 'enthalpy_atom'
    #config.label_str = 'Egap'
    #config.num_edges_max = 13_000
    config.init_lr = 1e-5

    return config

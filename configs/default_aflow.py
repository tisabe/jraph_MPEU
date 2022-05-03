"""Define the default hyperparameters for model and training."""

import ml_collections
from configs import default_mp as cfg

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = cfg.get_config() # inherit from default mp config

    config.data_file = 'aflow/graphs_cutoff_6A.db'
    config.selection = 'natoms<80'
    #config.label_str = 'enthalpy_formation_atom'
    config.label_str = 'Egap'
    config.num_edges_max = 13_000
    config.init_lr = 1e-5

    return config

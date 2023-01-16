"""Define the default hyperparameters for model and training."""

import ml_collections
from jraph_MPEU_configs.default_mp_test import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = get_config_super() # inherit from default mp config

    config.eval_every_steps = 1_000

    config.data_file = 'aflow/graphs_knn.db'
    config.selection = None
    config.label_str = 'enthalpy_formation_atom'
    config.num_edges_max = None

    return config

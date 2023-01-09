"""Define the default hyperparameters for model and training."""

import ml_collections
from jraph_MPEU_configs.default_mp import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = get_config_super() # inherit from default mp config

    config.data_file = 'aflow/graphs_knn_fix.db'
    config.label_str = 'enthalpy_formation_atom'
    #config.label_str = 'Egap'
    config.init_lr = 1e-4
    # remove outliers in formation enthalpy and other dft types
    config.selection = (
        "enthalpy_formation_atom<70,"
        "enthalpy_formation_atom>-10,"
        "dft_type=['PAW_PBE']")
    config.use_layer_norm = False

    return config

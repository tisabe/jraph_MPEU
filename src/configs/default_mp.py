"""Define the default hyperparameters for model and training."""

import ml_collections
from configs import default as cfg

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = cfg.get_config() # inherit from default config

    # data split settings
    config.data_file = 'matproj/mp_graphs.db'
    config.label_str = 'delta_e' # string to determine which label is used from the dataset
    config.val_frac = 0.1 # fraction of total data used for validation
    config.test_frac = 0.1 # fraction of total data used for testing

    # data selection parameters
    config.selection = 'fold>=0'

    # MPNN hyperparameters
    config.latent_size = 256
    config.max_input_feature_size = 100
    config.aggregation_message_type = 'mean'
    config.aggregation_readout_type = 'mean'
    # Node embedding parameters
    config.max_atomic_number = 90

    return config

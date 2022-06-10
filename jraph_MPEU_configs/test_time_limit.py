"""Define the default hyperparameters for model and training."""

import ml_collections
from jraph_MPEU_configs.default_test import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = get_config_super() # inherit from default config

    config.num_train_steps_max = 100_000
    config.time_limit = 0.01

    return config

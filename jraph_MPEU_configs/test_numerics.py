"""Define the default hyperparameters for model and training.
Test numerical stability with this config."""

import ml_collections
from jraph_MPEU_configs.default import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = get_config_super() # inherit from default config

    # Training hyperparameters
    config.num_train_steps_max = 20
    config.log_every_steps = 20
    config.eval_every_steps = 10
    config.checkpoint_every_steps = 10

    # data selection parameters
    config.limit_data = 2000

    return config

"""Define the default hyperparameters for model and training."""

import ml_collections
from jraph_MPEU_configs.default import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = get_config_super() # inherit from default config

    # Training hyperparameters
    config.num_train_steps_max = 10_000
    config.log_every_steps = 500
    config.eval_every_steps = 500
    config.early_stopping_steps = 1_000
    config.checkpoint_every_steps = 1_000
    config.restore = False

    # data selection parameters
    config.limit_data = 200

    config.log_to_file = False # if logging should go to file if true or console if false
    return config

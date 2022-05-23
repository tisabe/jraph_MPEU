"""Define the default hyperparameters for model and training."""

import ml_collections
from jraph_MPEU_configs.default import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = get_config_super()  # inherit from default config

    # Optimizer
    config.schedule = 'cosine_decay'
    config.init_lr = 1e-3
    config.transition_steps = 1_000_000

    return config

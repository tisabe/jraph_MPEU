
'''Define the hyperparameters for model and training, to test numerical stability.
'''

import ml_collections
from configs import default as cfg

def get_config() -> ml_collections.ConfigDict():
    '''Get hyperparameter configuration. Returns a ml_collections.ConfigDict() object.
    '''
    config = cfg.get_config() # inherit from default config
    
    # Training hyperparameters
    config.num_train_steps_max = 100
    config.log_every_steps = 5_000
    config.eval_every_steps = 10
    
    # data selection parameters
    config.limit_data = 2000
    
    return config

'''Define the default hyperparameters for model and training.
'''

import ml_collections
from configs import default_test as cfg

def get_config() -> ml_collections.ConfigDict():
    '''Get hyperparameter configuration. Returns a ml_collections.ConfigDict() object.
    '''
    config = cfg.get_config() # inherit from default config
    
    config.num_train_steps_max = 100_000
    config.time_limit = 0.01
    return config
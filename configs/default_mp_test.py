
'''Define the default hyperparameters for model and training.
'''

import ml_collections
from configs import default_mp as cfg

def get_config() -> ml_collections.ConfigDict():
    '''Get hyperparameter configuration. Returns a ml_collections.ConfigDict() object.
    '''
    config = cfg.get_config() # inherit from default mp config
    
    config.num_train_steps_max = 10_000
    config.log_every_steps = 5_000
    config.eval_every_steps = 10_000
    config.early_stopping_steps = 100_000
    config.checkpoint_every_steps = 50_000
    # data split settings
    config.label_str = 'band_gap' # string to determine which label is used from the dataset
    
    # data selection parameters
    config.selection = 'fold=1,band_gap>0'
    config.limit_data = 2000

    # MPNN hyperparameters
    config.latent_size = 64
    # Node embedding parameters
    config.max_atomic_number = 90
    
    return config

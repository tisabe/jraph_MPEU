'''Define the default hyperparameters for model and training.
'''

import ml_collections
from configs import default_mp as cfg

def get_config() -> ml_collections.ConfigDict():
    '''Get hyperparameter configuration. Returns a ml_collections.ConfigDict() object.
    '''
    config = cfg.get_config() # inherit from default mp config
    
    config.data_file = 'aflow/graphs_cutoff_6A.db'
    config.selection = 'natoms<100'
    config.label_str = 'enthalpy_formation_atom'
    
    return config

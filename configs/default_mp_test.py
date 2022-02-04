
'''Define the default hyperparameters for model and training.
'''

import ml_collections

def get_config() -> ml_collections.ConfigDict():
    '''Get hyperparameter configuration. Returns a ml_collections.ConfigDict() object.
    '''
    config = ml_collections.ConfigDict()
    
    # Optimizer
    config.optimizer = 'adam'
    config.schedule = 'exponential_decay'
    config.init_lr = 5e-4 # initial learning rate
    # parameters for exponential schedule
    config.transition_steps = 100_000
    config.decay_rate = 0.96

    config.loss_type = 'MSE'

    # Training hyperparameters
    config.batch_size = 32
    config.num_train_steps_max = 10_000
    config.log_every_steps = 5_000
    config.eval_every_steps = 10_000
    config.early_stopping_steps = 100_000
    config.checkpoint_every_steps = 50_000
    # data split settings
    config.data_file = 'matproj/mp_graphs.db'
    config.label_str = 'delta_e' # string to determine which label is used from the dataset
    config.val_frac = 0.1 # fraction of total data used for validation
    config.test_frac = 0.1 # fraction of total data used for testing

    # MPNN hyperparameters
    config.message_passing_steps = 3
    config.latent_size = 64
    config.hk_init = None
    config.max_input_feature_size = 100
    config.aggregation_message_type = 'mean'
    config.aggregation_readout_type = 'mean'
    # Edge embedding parameters
    config.k_max = 150
    config.delta = 0.1
    config.mu_min = 0.0
    # Node embedding parameters
    config.max_atomic_number = 90

    # Logging options
    config.log_to_file = False # if logging should go to file if true or console if false
    return config

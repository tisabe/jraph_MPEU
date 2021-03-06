"""Define the default hyperparameters for cross validation on aflow data."""

import ml_collections

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = ml_collections.ConfigDict()

    # rng init
    config.seed = [42]

    # Optimizer
    config.optimizer = ['adam']
    config.schedule = ['exponential_decay']
    config.init_lr = [1e-3, 1e-4, 1e-5] # initial learning rate
    # parameters for exponential schedule
    config.transition_steps = [100_000]
    config.decay_rate = [0.92, 0.96, 1.0]

    config.loss_type = ['MSE']

    # Training hyperparameters
    config.batch_size = [32]
    config.num_train_steps_max = [2_000_000]
    config.log_every_steps = [10_000]
    config.eval_every_steps = [50_000]
    config.early_stopping_steps = [1_000_000]
    config.checkpoint_every_steps = [500_000]
    config.restore = [False] # whether to restore from previous checkpoint
    # data split settings
    config.data_file = ['aflow/graphs_knn.db']
    config.label_str = ['enthalpy_formation_atom']
    config.val_frac = [0.1] # fraction of total data used for validation
    config.test_frac = [0.1] # fraction of total data used for testing

    # data selection parameters
    config.selection = [None]
    config.limit_data = [None]
    config.num_edges_max = [None]

    # MPNN hyperparameters
    config.message_passing_steps = [2, 3, 4]
    config.latent_size = [64, 128, 256]
    config.hk_init = [None]
    config.max_input_feature_size = [100]
    config.aggregation_message_type = ['mean']
    config.aggregation_readout_type = ['mean']
    # Edge embedding parameters
    config.k_max = [150]
    config.delta = [0.1]
    config.mu_min = [0.0]
    # Node embedding parameters
    config.max_atomic_number = [90]

    # Logging options
    config.log_to_file = [False] # if logging should go to file if true or console if false
    return config

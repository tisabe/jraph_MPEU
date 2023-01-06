"""Define the default hyperparameters for random architecture search on aflow
data."""

import ml_collections

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = ml_collections.ConfigDict()

    # rng init
    config.base_data_seed = [42]

    # Optimizer
    config.optimizer = ['adam']
    config.schedule = ['exponential_decay']
    config.init_lr = [1e-4] # initial learning rate
    # parameters for exponential schedule
    config.transition_steps = [100_000]
    config.decay_rate = [1.0]

    config.loss_type = ['MSE']

    # Training hyperparameters
    config.batch_size = [32]
    config.num_train_steps_max = [100_000]
    config.log_every_steps = [10_000]
    config.eval_every_steps = [50_000]
    config.early_stopping_steps = [1_000_000]
    config.checkpoint_every_steps = [100_000]
    config.num_checkpoints = [1]
    config.restore = [False] # whether to restore from previous checkpoint
    # data split settings
    config.data_file = ['aflow/egap_full_graphs.db']
    config.label_str = ['Egap']
    config.label_type = ['scalar']  # or 'class', also changes the loss function
    config.val_frac = [0.1] # fraction of total data used for validation
    config.test_frac = [0.1] # fraction of total data used for testing

    # data selection parameters
    # remove outliers in formation enthalpy and other dft types
    config.selection = [("dft_type=['PAW_PBE']")]
    config.limit_data = [None]
    config.num_edges_max = [None]

    # MPNN hyperparameters
    config.message_passing_steps = [1, 2, 3]
    config.latent_size = [32, 64, 128, 256]
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
    config.extra_mlp = [False]
    config.dropout_rate = [0.0]

    # Logging options
    config.log_to_file = [False] # if logging should go to file if true or console if false
    return config
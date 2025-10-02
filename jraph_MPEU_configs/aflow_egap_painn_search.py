"""Define the default hyperparameters for aflow data."""

import ml_collections

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = ml_collections.ConfigDict()

    # rng init
    config.seed_splits = [42]
    config.seed_datareader = [42]
    config.seed_weights = [42]
    config.shuffle_val_seed = [-1]

    # Optimizer
    config.optimizer = ['adam']
    config.schedule = ['exponential_decay']
    config.init_lr = [1e-5, 1e-4, 1e-3] # initial learning rate
    # parameters for exponential schedule
    config.transition_steps = [100_000]
    config.decay_rate = [0.96, 0.98, 1.0]

    config.loss_type = ['MSE']

    # Training hyperparameters
    config.batch_size = [50, 100]
    config.num_train_steps_max = [10_000_000]
    config.log_every_steps = [10_000]
    config.eval_every_steps = [50_000]
    config.early_stopping_steps = [1_000_000]
    config.checkpoint_every_steps = [100_000]
    config.num_checkpoints = [1]
    # data split settings
    config.data_file = ['databases/aflow/graphs_12knn_vec.db']
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
    config.model_str = ['PaiNN']
    config.cutoff_radius = [6.]
    config.message_passing_steps = [1, 3, 5]
    config.latent_size = [128, 256, 512]
    config.max_input_feature_size = [100]
    config.aggregation_message_type = ['sum', 'mean']
    config.aggregation_readout_type = ['mean']
    # Node embedding parameters
    config.max_atomic_number = [90]

    # Logging options
    config.log_to_file = [False] # if logging should go to file if true or console if false
    return config

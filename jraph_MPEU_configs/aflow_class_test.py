"""Define the hyperparameters for model and training."""

import ml_collections

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = ml_collections.ConfigDict()

    # rng init
    config.seed = 50

    # Optimizer
    config.optimizer = 'adam'
    config.schedule = 'exponential_decay'
    config.init_lr = 5e-4 # initial learning rate
    # parameters for exponential schedule
    config.transition_steps = 100_000
    config.decay_rate = 0.96

    # Training hyperparameters
    config.batch_size = 32
    config.num_train_steps_max = 10_000
    config.log_every_steps = 1_000
    config.eval_every_steps = 5_000
    config.early_stopping_steps = 1_000_000
    config.checkpoint_every_steps = 10_000
    config.num_checkpoints = 1  # number of checkpoints to keep
    # data split settings
    config.data_file = 'aflow/egap_full_graphs.db'
    config.label_str = 'Egap' # string to determine which label is used from the dataset
    config.val_frac = 0.1 # fraction of total data used for validation
    config.test_frac = 0.1 # fraction of total data used for testing

    # type of label
    config.label_type = 'class'  # or 'class', also changes the loss function
    config.egap_cutoff = 0.0  # below which band structures are counted as metals

    # data selection parameters
    config.selection = None
    config.limit_data = 2000
    config.num_edges_max = None

    # MPNN hyperparameters
    config.message_passing_steps = 3
    config.latent_size = 64
    config.hk_init = None
    config.max_input_feature_size = 100
    config.aggregation_message_type = 'mean'
    config.aggregation_readout_type = 'mean'
    config.global_readout_mlp_layers = 2
    config.mlp_depth = 2
    config.activation_name = 'shifted_softplus'
    # Edge embedding parameters
    config.k_max = 150
    config.delta = 0.1
    config.mu_min = 0.0
    # Node embedding parameters
    config.max_atomic_number = 5
    # Regularization parameters
    config.use_layer_norm = False
    config.dropout_rate = 0.0

    # Logging options
    config.log_to_file = False # if logging should go to file if true or console if false
    return config

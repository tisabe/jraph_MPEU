"""Define the default hyperparameters for model and training."""

import ml_collections

def get_config() -> ml_collections.ConfigDict():
    """Get hyperparameter configuration.
    Returns a ml_collections.ConfigDict() object."""
    config = ml_collections.ConfigDict()

    # rng init
    config.seed = 42

    # Optimizer
    config.optimizer = 'adam'
    config.schedule = 'exponential_decay'
    config.init_lr = 1e-4 # initial learning rate
    # parameters for exponential schedule
    config.transition_steps = 100_000
    config.decay_rate = 0.96

    # Training hyperparameters
    config.batch_size = 32
    config.num_train_steps_max = 10_000_000
    config.log_every_steps = 10_000
    config.eval_every_steps = 50_000
    config.early_stopping_steps = 1_000_000
    config.checkpoint_every_steps = 100_000
    config.num_checkpoints = 1  # number of checkpoints to keep
    # data split settings
    config.data_file = 'aflow/graphs_all_12knn.db'
    config.label_str = 'enthalpy_formation_atom'
    config.val_frac = 0.1 # fraction of total data used for validation
    config.test_frac = 0.1 # fraction of total data used for testing

    # type of label
    config.label_type = 'scalar'  # or 'class', also changes the loss function

    # data selection parameters
    config.selection = (
        "enthalpy_formation_atom<70,"
        "enthalpy_formation_atom>-10,"
        "dft_type=['PAW_PBE'],"
        "ldau_type=0.0")
    config.limit_data = None
    config.num_edges_max = None

    # MPNN hyperparameters
    config.message_passing_steps = 3
    config.latent_size = 256
    config.hk_init = None
    config.max_input_feature_size = 100
    config.aggregation_message_type = 'sum'
    config.aggregation_readout_type = 'mean'
    config.global_readout_mlp_layers = 0
    config.mlp_depth = 1
    config.activation_name = 'shifted_softplus'
    config.max_atomic_number = 100
    # Regularization parameters
    config.use_layer_norm = False
    config.dropout_rate = 0

    # Logging options
    config.log_to_file = False # if logging should go to file if true or console if false
    return config

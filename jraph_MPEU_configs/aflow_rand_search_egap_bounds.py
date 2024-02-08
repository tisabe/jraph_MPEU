"""Define the default hyperparameters for random architecture search on aflow
data. This config tests the upper and lower bounds for hyper parameters for
later use in a bigger search"""

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
    config.init_lr = [2e-5, 5e-4] # initial learning rate
    # parameters for exponential schedule
    config.transition_steps = [100_000]
    config.decay_rate = [0.96, 1.0]

    config.loss_type = ['MSE']

    # Training hyperparameters
    config.batch_size = [16, 128]
    config.num_train_steps_max = [10_000_000]
    config.log_every_steps = [10_000]
    config.eval_every_steps = [50_000]
    config.early_stopping_steps = [1_000_000]
    config.checkpoint_every_steps = [100_000]
    config.num_checkpoints = [1]
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
    config.model_str = ['MPEU']
    config.message_passing_steps = [1, 7]
    config.latent_size = [64, 512]
    config.hk_init = [None]
    config.max_input_feature_size = [100]
    config.aggregation_message_type = ['mean']
    config.aggregation_readout_type = ['mean']
    config.global_readout_mlp_layers = [0, 4]
    config.mlp_depth = [2, 4]
    config.activation_name = ['shifted_softplus', 'relu', 'swish']
    # Edge embedding parameters
    config.k_max = [150]
    config.delta = [0.1]
    config.mu_min = [0.0]
    # Node embedding parameters
    config.max_atomic_number = [90]
    config.use_layer_norm = [False, True]
    config.dropout_rate = [0.0, 0.5]

    # Logging options
    config.log_to_file = [False] # if logging should go to file if true or console if false
    return config

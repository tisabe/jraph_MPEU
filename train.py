import os
from typing import NamedTuple, Callable, Dict, Iterable, Sequence

import logging
import jax
import jax.numpy as jnp
from flax.training import train_state
import jraph
import ml_collections
import numpy as np
import optax
import haiku as hk

# import custom functions
from graph_net_fn import net_fn
from utils import *
import config as config_globals # TODO: switch from global parameters
from input_pipeline import get_datasets

'''
class TrainState(NamedTuple):
    apply_fn: hk.Transformed # graph net function
    params: hk.Params # haiku weight parameters
    opt_state: optax.OptState # optax optimizer state
    opt_update: optax.Updates # optax optimizer update
    step: int
'''

def create_model(config: ml_collections.ConfigDict):
    '''Return a function that applies the graph model.'''
    # first an interface between globals in config and passed config. 
    # TODO: get rid of globals
    config_globals.LABEL_SIZE = 1
    config_globals.N_HIDDEN_C = config.latent_size
    config_globals.NUM_MP_LAYERS = config.message_passing_steps
    config_globals.AVG_MESSAGE = config.avg_aggregation_message
    config_globals.AVG_READOUT = config.avg_aggregation_readout
    return net_fn


def create_optimizer(
    config: ml_collections.ConfigDict) -> optax.GradientTransformation:
    if config.schedule == 'exponential_decay':
        lr = optax.exponential_decay(
            init_value=config.init_lr, 
            transition_steps=config.transition_steps, 
            decay_rate=config.decay_rate)

    if config.optimizer == 'adam':
        return optax.adam(learning_rate=lr)
    raise ValueError(f'Unsupported optimizer: {config.optimizer}.')


@jax.jit
def train_step(
    state: train_state.TrainState, graphs: jraph.GraphsTuple
) -> Tuple[train_state.TrainState, float]:
    '''Perform one update step over the batch of graphs. 
    Returns a new TrainState and the loss MAE over the batch.'''
    return state, 0.0


@jax.jit
def evaluate_step(
    state: train_state.TrainState, graphs: jraph.GraphsTuple
) -> float:
    return 0.0


def evaluate_model(
    state: train_state.TrainState,
    datasets: Dict[str, Sequence[jraph.GraphsTuple]],
    splits: Iterable[str]
) -> Dict[str, float]:
    return {}


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str
) -> train_state.TrainState:
    # Get datasets, organized by split.
    logging.info('Loading datasets.')
    datasets = get_datasets(config)

    # Create and initialize network.
    logging.info('Initializing network.')
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    init_graphs = next(datasets['train'])
    init_graphs = replace_globals(init_graphs)
    #print(init_graphs)
    net_fn = create_model(config)
    net = hk.without_apply_rng(hk.transform(net_fn))
    params = net.init(rng, init_graphs) # create weights etc. for the model

    # Create the optimizer
    tx = create_optimizer(config)
    #opt_init, opt_update = create_optimizer(config)
    #opt_state = opt_init(params)

    # Create the training state
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)
    
    # Set up checkpointing of the model.
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    # start at step 1
    initial_step = int(state.step) + 1
    # TODO: get some framework for automatic checkpoint restoring

    # Begin training loop.
    logging.info('Starting training.')
    for step in range(initial_step, config.num_train_steps_max + 1):

        # Split PRNG key, to ensure different 'randomness' for every step.
        rng, dropout_rng = jax.random.split(rng)

        # Perform a training step
        graphs = next(datasets['train'])
        state, loss = train_step(state, graphs)




    return 0
    


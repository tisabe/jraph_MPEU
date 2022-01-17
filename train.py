import logging
import jax
import jax.numpy as jnp
import jraph
import ml_collections
from typing import NamedTuple, Callable, Dict, Iterable, Sequence
import numpy as np
import optax
import haiku as hk

# import custom functions
from graph_net_fn import net_fn
from utils import *
import config as config_globals # TODO: switch from global parameters


class TrainState(NamedTuple):
    net: hk.Transformed # graph net function
    params: hk.Params # haiku weight parameters
    opt_state: optax.OptState # optax optimizer state
    opt_update: optax.Updates # optax optimizer update


def create_model(config: ml_collections.ConfigDict):
    '''Return a function that applies the graph model.'''
    # first an interface between globals in config and passed config. 
    # TODO: get rid of globals
    config_globals.LABEL_SIZE = 1
    config_globals.N_HIDDEN_C = config.latent_size
    config_globals.NUM_MP_LAYERS = config.message_passing_steps
    config_globals.AVG_MESSAGE = config.avg_aggregation_message
    config_globals.AVG_READOUT = config.avg_aggregation_readout
    return net_nf


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
    state: TrainState, graphs: jraph.GraphsTuple
) -> Tuple[TrainState, float]:
    '''Perform one update step over the batch of graphs. 
    Returns a new TrainState and the loss MAE over the batch.'''
    return state, 0.0


@jax.jit
def evaluate_step(
    state: TrainState, graphs: jraph.GraphsTuple
) -> float:
    return 0.0


def evaluate_model(
    state: TrainState,
    datasets: Dict[str, Sequence[jraph.GraphsTuple]],
    splits: Iterable[str]
) -> Dict[str, float]:
    return {}


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str
) -> TrainState:
    # Get datasets, organized by split.
    logging.info('Loading datasets.')

    return 0
    


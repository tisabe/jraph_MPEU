import os
from typing import NamedTuple, Callable, Dict, Iterable, Sequence

from absl import logging
import jax
import jax.numpy as jnp
from flax.training import train_state
import jraph
import ml_collections
import numpy as np
import optax
import haiku as hk
import pickle

# import custom functions
from graph_net_fn import net_fn
from utils import *
import config as config_globals # TODO: switch from global parameters
from input_pipeline import get_datasets
from input_pipeline import DataReader

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
            decay_rate=config.decay_rate,
            staircase=True)

    if config.optimizer == 'adam':
        return optax.adam(learning_rate=lr)
    raise ValueError(f'Unsupported optimizer: {config.optimizer}.')


@jax.jit
def train_step(
    state: train_state.TrainState, graphs: jraph.GraphsTuple
) -> Tuple[train_state.TrainState, float]:
    '''Perform one update step over the batch of graphs. 
    Returns a new TrainState and the loss MAE over the batch.'''

    def loss_fn(params, graphs):
        curr_state = state.replace(params=params)

        labels = graphs.globals
        graphs = replace_globals(graphs)

        mask = get_valid_mask(labels, graphs)
        pred_graphs = state.apply_fn(curr_state.params, graphs)
        predictions = pred_graphs.globals
        labels = jnp.expand_dims(labels, 1)
        abs_diff = jnp.abs((predictions - labels)*mask)
        # TODO: make different loss functions available in config
        loss = jnp.sum(abs_diff)
        mean_loss = loss / jnp.sum(mask)

        return mean_loss, (loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (mean_loss, (loss)), grads = grad_fn(state.params, graphs)
    # update the params with gradient
    state = state.apply_gradients(grads=grads)

    return state, mean_loss


@jax.jit
def evaluate_step(
    state: train_state.TrainState, graphs: jraph.GraphsTuple
) -> float:
    '''Calculate the mean loss for a batch of graphs.'''
    labels = graphs.globals
    graphs = replace_globals(graphs)

    mask = get_valid_mask(labels, graphs)
    pred_graphs = state.apply_fn(state.params, graphs)
    predictions = pred_graphs.globals
    labels = jnp.expand_dims(labels, 1)
    abs_diff = jnp.abs((predictions - labels)*mask)
    # TODO: make different loss functions available in config
    loss = jnp.sum(abs_diff)
    mean_loss = loss / jnp.sum(mask)
    return mean_loss


def evaluate_model(
    state: train_state.TrainState,
    datasets: Dict[str, Sequence[jraph.GraphsTuple]],
    splits: Iterable[str]
) -> Dict[str, float]:

    eval_loss = {}
    for split in splits:
        mean_loss_sum = 0.0
        batch_count = 0

        # the following is a hack at best, but it works
        batch_size = datasets[split].batch_size
        data = datasets[split].data
        reader_new = DataReader(data=data, 
            batch_size=batch_size, repeat=False, key=None)
        
        # loop over all graphs in split dataset 
        # (repeat has to be disabled in this split)
        for graphs in reader_new:
            mean_loss = evaluate_step(state, graphs)
            mean_loss_sum += mean_loss
            batch_count += 1
        
        eval_loss[split] = mean_loss_sum / batch_count
    
    return eval_loss


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str
) -> train_state.TrainState:
    # Initialize rng
    rng = jax.random.PRNGKey(42)

    # Get datasets, organized by split.
    rng, data_rng = jax.random.split(rng) # split up rngs for deterministic results
    logging.info('Loading datasets.')
    datasets = get_datasets(config, data_rng)

    # Create and initialize network.
    logging.info('Initializing network.')
    rng, init_rng = jax.random.split(rng)
    init_graphs = next(datasets['train'])
    init_graphs = replace_globals(init_graphs) # initialize globals in graph to zero
    #print(init_graphs)
    net_fn = create_model(config)
    net = hk.without_apply_rng(hk.transform(net_fn))
    # TODO: find which initialization is used here and in PBJ
    params = net.init(init_rng, init_graphs) # create weights etc. for the model
    
    # Create the optimizer
    tx = create_optimizer(config)
    logging.info(f'Init_lr: {config.init_lr}')
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

    # Make a loss queue to compare with earlier losses
    loss_queue = []
    params_queue = []
    best_params = None

    # set up saving of losses
    splits = ['train', 'validation', 'test']
    loss_dict = {}
    for split in splits:
        loss_dict[split] = []

    # Begin training loop.
    logging.info('Starting training.')
    time_logger = Time_logger(config)

    for step in range(initial_step, config.num_train_steps_max + 1):

        # Split PRNG key, to ensure different 'randomness' for every step.
        rng, dropout_rng = jax.random.split(rng)

        # Perform a training step
        graphs = next(datasets['train'])
        state, loss = train_step(state, graphs)

        # Log periodically
        is_last_step = (step == config.num_train_steps_max - 1)
        if step % config.log_every_steps == 0 or is_last_step:
            time_logger.log_eta(step)
        
        # evaluate model on train, test and validation data
        if step % config.eval_every_steps == 0 or is_last_step:
            eval_loss = evaluate_model(state, datasets, splits)
            for split in splits:
                logging.info(f'MAE {split}: {eval_loss[split]}')
                loss_dict[split].append(eval_loss[split])
            
            loss_queue.append(eval_loss['validation'])
            params_queue.append(state.params)
            # only test for early stopping after the first interval
            if step > config.early_stopping_steps:
                # stop if new loss higher than loss at beginning of interval
                if eval_loss['validation'] > loss_queue[0]:
                    break
                else:
                    # otherwise delete the element at beginning of queue
                    loss_queue.pop(0)
                    params_queue.pop(0)
        
    
    
    # save parameters of best model
    index = np.argmin(loss_queue)
    params = params_queue[index]
    with open((workdir+'/params.pickle'), 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    for split in splits:
        np.savetxt(f'{workdir}/{split}_loss.csv', 
            np.array(loss_dict[split]), delimiter=",")

    return 0
    


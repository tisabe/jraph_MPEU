import os
from typing import NamedTuple, Callable, Dict, Iterable, Sequence, Tuple

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
from models import GNN
from utils import *
from input_pipeline import get_datasets
from input_pipeline import DataReader


def make_result_csv(x, y, path):
    '''Print predictions x versus labels y in a csv at path.'''
    dict_res = {'x': np.array(x).flatten(), 'y': np.array(y).flatten()}
    df = pandas.DataFrame(data=dict_res)
    df.to_csv(path)


def get_globals(graphs: Sequence[jraph.GraphsTuple]
) -> Sequence[float]:
    labels = []
    for graph in graphs:
        labels.append(float(graph.globals))

    return labels


def get_labels_original(graphs, labels, label_str):
    '''Wrapper function to get original energies,
    to make it compatible with non-QM9 datasets.'''
    return get_original_energies_QM9(graphs, labels, label_str)


def create_model(config: ml_collections.ConfigDict):
    '''Return a function that applies the graph model.'''
    return GNN(config)


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

def get_diff_fn(config: ml_collections.ConfigDict
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[float, float]]:
    '''Return difference function depending on str argument in config.loss_type.
    The difference function defines how loss is calculated from label and prediction.'''
    if config.loss_type == 'MSE':
        def diff_fn(labels, predictions, mask):
            labels = jnp.expand_dims(labels, 1)
            sq_diff = jnp.square((predictions - labels)*mask)
            # TODO: make different loss functions available in config
            loss = jnp.sum(sq_diff)
            mean_loss = loss / jnp.sum(mask)

            return mean_loss, (loss)
        return loss_fn
    elif config.loss_type == 'MAE':
        def diff_fn(labels, predictions, mask):
            labels = jnp.expand_dims(labels, 1)
            diff = jnp.abs((predictions - labels)*mask)
            # TODO: make different loss functions available in config
            loss = jnp.sum(diff)
            mean_loss = loss / jnp.sum(mask)

            return mean_loss, (loss)
        return loss_fn      
    raise ValueError(f'Unsupported loss type: {config.loss_type}.')

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
        sq_diff = jnp.square((predictions - labels)*mask)
        # TODO: make different loss functions available in config
        loss = jnp.sum(sq_diff)
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
    sq_diff = jnp.square((predictions - labels)*mask)
    # TODO: make different loss functions available in config
    loss = jnp.sum(sq_diff)
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


def predict_split(
    state: train_state.TrainState,
    dataset_raw: Sequence[jraph.GraphsTuple],
    config: ml_collections.ConfigDict
) -> Sequence[float]:

    preds = np.array([])
    reader_new = DataReader(data=dataset_raw, 
        batch_size=config.batch_size, repeat=False, key=None)

    for graphs in reader_new:
        labels = graphs.globals
        graphs = replace_globals(graphs)

        mask = get_valid_mask(labels, graphs)
        pred_graphs = state.apply_fn(state.params, graphs)
        preds_batch = pred_graphs.globals
        # throw away all padding labels
        preds_batch_valid = preds_batch[mask]
        # update predictions list
        preds = np.concatenate((preds, preds_batch_valid), axis=0)
    
    return preds


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str
) -> train_state.TrainState:
    # Initialize rng
    rng = jax.random.PRNGKey(42)

    # Get datasets, organized by split.
    rng, data_rng = jax.random.split(rng) # split up rngs for deterministic results
    logging.info('Loading datasets.')
    datasets, datasets_raw, mean, std = get_datasets(config, data_rng)

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
                logging.info(f'MSE {split}: {eval_loss[split]}')
                loss_dict[split].append([step, eval_loss[split]])
            
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

    # save predictions of the best model
    best_state = state.replace(params=params) # restore the state with best params
    #pred_dict = predict(best_state, datasets_raw, splits)
    
    for split in splits:
        loss_split = np.array(loss_dict[split])
        # convert loss column to eV
        loss_split[:,1] = loss_split[:,1]*std
        # save the loss curves
        np.savetxt(f'{workdir}/{split}_loss.csv', 
            np.array(loss_split), delimiter=",")

        # save the predictions and labels
        #labels = get_globals(datasets[split].data)
        labels = get_globals(datasets_raw[split])
        preds = predict_split(best_state, datasets_raw[split], config)
        #preds = scale_targets_config(datasets_raw[split], preds, mean, std, config)
        #preds = get_labels_original(datasets_raw[split], preds, config.label_str)
        labels = np.array(labels)
        preds = np.array(preds)
        make_result_csv(
            labels, preds, 
            f'{workdir}/{split}_post.csv')


    return best_state
    


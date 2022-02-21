import os
from typing import (
    NamedTuple, 
    Callable, 
    Dict, 
    Iterable, 
    Sequence, 
    Tuple,
    Optional
)
from absl import logging
import jax
import jax.numpy as jnp
from flax.training import train_state
import flax.training.checkpoints as ckpt
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
    elif config.schedule == 'cosine_decay':
        lr = optax.cosine_decay_schedule(
            init_value=config.init_lr, 
            decay_steps=1e6)
    else:
        raise ValueError(f'Unsupported schedule: {config.schedule}.')

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


def evaluate_split(
    state: train_state.TrainState,
    graphs: Sequence[jraph.GraphsTuple],
    batch_size = int
) -> float:
    '''Return mean loss for all graphs in graphs.'''
    mean_loss_sum = 0.0
    batch_count = 0

    reader = DataReader(data=graphs, 
        batch_size=batch_size, repeat=False)
    
    for graph_batch in reader:
        mean_loss = evaluate_step(state, graph_batch)
        mean_loss_sum += mean_loss
        batch_count += 1
    
    return mean_loss_sum / batch_count


def evaluate_model(
    state: train_state.TrainState,
    datasets: Dict[str, Iterable[jraph.GraphsTuple]],
    splits: Iterable[str]
) -> Dict[str, float]:
    '''Return mean loss for every split in splits.'''
    eval_loss = {}
    for split in splits:
        eval_loss[split] = evaluate_split(state, 
            datasets[split].data, datasets[split].batch_size)
    
    return eval_loss


def predict_split(
    state: train_state.TrainState,
    dataset_raw: Sequence[jraph.GraphsTuple],
    config: ml_collections.ConfigDict
) -> Sequence[float]:

    preds = np.array([])
    reader_new = DataReader(data=dataset_raw, 
        batch_size=config.batch_size, repeat=False)

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


def init_state(
    config: ml_collections.ConfigDict,
    init_graphs: jraph.GraphsTuple
) -> train_state.TrainState:
    '''Initialize a TrainState object using hyperparameters in config,
    and the init_graphs. This is a representative batch of graphs.'''
    # Initialize rng
    rng = jax.random.PRNGKey(config.seed)

    # Create and initialize network.
    logging.info('Initializing network.')
    rng, init_rng = jax.random.split(rng)
    init_graphs = replace_globals(init_graphs) # initialize globals in graph to zero
    
    net_fn = create_model(config)
    net = hk.without_apply_rng(hk.transform(net_fn))
    # TODO: check changing initializer
    params = net.init(init_rng, init_graphs) # create weights etc. for the model
    
    # Create the optimizer
    tx = create_optimizer(config)
    logging.info(f'Init_lr: {config.init_lr}')

    # Create the training state
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)

    return state


def restore_checkpoint(state, checkpoint_dir):
    with open((checkpoint_dir+'/params.pickle'), 'rb') as handle:
        state_dict = pickle.load(handle)
    return state.replace(params=state_dict['params'], step=state_dict['step'])
    

def save_checkpoint(state, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    state_dict = {'params': state.params, 'step': state.step}
    with open((checkpoint_dir+'/params.pickle'), 'wb') as handle:
        pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def restore_loss_curve(dir, splits, std):
    loss_dict = {}
    for split in splits:
        loss_split = np.loadtxt(f'{dir}/{split}_loss.csv', 
            delimiter=',', ndmin=2)
        loss_split[:,1] = loss_split[:,1]/std
        loss_dict[split] = loss_split.tolist()
    return loss_dict

def save_loss_curve(loss_dict, dir, splits, std):
    for split in splits:
        loss_split = np.array(loss_dict[split])
        # convert loss column to eV
        loss_split[:,1] = loss_split[:,1]*std
        # save the loss curves
        np.savetxt(f'{dir}/{split}_loss.csv', 
            np.array(loss_split), delimiter=',')


def train(
    config: ml_collections.ConfigDict,
    datasets: Dict[str, Sequence[jraph.GraphsTuple]],
    workdir: Optional[str] = None
) -> Tuple[train_state.TrainState, float]:
    '''Train a model using training data in dataset and validation data 
    in datasets for early stopping and model selection.
    
    The globals of training and validation graphs need to be normalized,
    and the loss will be on normalized errors.'''
    
    reader_train = DataReader(datasets['train'], config.batch_size, 
        repeat=True, seed=config.seed)
    init_graphs = next(reader_train)

    state = init_state(config, init_graphs)
    
    if workdir is not None:
        # Set up checkpointing of the model.
        checkpoint_dir = os.path.join(workdir, 'checkpoints')
    # start at step 1 (or state.step + 1 if state was restored)
    initial_step = int(state.step) + 1
    # TODO: get some framework for automatic checkpoint restoring

    loss_queue = []
    params_queue = []
    best_params = None
    
    logging.info('Starting training.')
    for step in range(initial_step, config.num_train_steps_max + 1):
        # Perform a training step
        graphs = next(reader_train)
        state, loss = train_step(state, graphs)

        is_last_step = (step == config.num_train_steps_max - 1)
        # evaluate model on train, test and validation data
        if step % config.eval_every_steps == 0 or is_last_step:
            eval_loss = evaluate_split(state, datasets['validation'],
                config.batch_size)
            logging.info(f'validation MSE: {eval_loss}')
            
            loss_queue.append(eval_loss)
            params_queue.append(state.params)
            # only test for early stopping after the first interval
            if step > config.early_stopping_steps:
                # stop if new loss higher than loss at beginning of interval
                if eval_loss > loss_queue[0]:
                    break
                else:
                    # otherwise delete the element at beginning of queue
                    loss_queue.pop(0)
                    params_queue.pop(0)

    # save parameters of best model
    index = np.argmin(loss_queue)
    # get lowest validation loss
    min_loss = loss_queue[index]
    params = params_queue[index]
    if workdir is not None:
        with open((workdir+'/params.pickle'), 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save predictions of the best model
    best_state = state.replace(params=params) # restore the state with best params
    
    return best_state, min_loss

    
def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str
) -> train_state.TrainState:
    
    logging.info('Loading datasets.')
    datasets, datasets_raw, mean, std = get_datasets(config)
    logging.info(f'Number of node classes: {config.max_atomic_number}')

    init_graphs = next(datasets['train'])
    init_graphs = replace_globals(init_graphs) # initialize globals in graph to zero
    
    # Create the training state
    state = init_state(config, init_graphs)
    
    # Set up checkpointing of the model.
    checkpoint_dir = os.path.join(workdir, 'checkpoints')

    # set up saving of losses
    splits = ['train', 'validation', 'test']
    loss_dict = {}
    for split in splits:
        loss_dict[split] = []

    if config.restore:
        state = restore_checkpoint(state, checkpoint_dir)
        loss_dict = restore_loss_curve(checkpoint_dir, splits, std)

    # start at step 1 (or state.step + 1 if state was restored)
    initial_step = int(state.step) + 1
    
    # Make a loss queue to compare with earlier losses
    loss_queue = []
    params_queue = []
    best_params = None

    # Begin training loop.
    logging.info('Starting training.')
    time_logger = Time_logger(config)

    for step in range(initial_step, config.num_train_steps_max + 1):
        # Perform a training step
        graphs = next(datasets['train'])
        state, loss = train_step(state, graphs)

        # Log periodically
        is_last_step = (step == config.num_train_steps_max)
        if step % config.log_every_steps == 0:
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
                    logging.info('Stopping early.')
                    break
                else:
                    # otherwise delete the element at beginning of queue
                    loss_queue.pop(0)
                    params_queue.pop(0)
        
        # Checkpoint model, if required
        if step % config.checkpoint_every_steps == 0 or is_last_step:
            save_checkpoint(state, checkpoint_dir)
            # save the loss curves
            save_loss_curve(loss_dict, checkpoint_dir, splits, std)
        if step==config.num_train_steps_max:
            logging.info('Reached maximum number of steps without early stopping.')

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
        preds = scale_targets_config(datasets_raw[split], preds, mean, std, config)
        #preds = get_labels_original(datasets_raw[split], preds, config.label_str)
        labels = np.array(labels)
        preds = np.array(preds)
        make_result_csv(
            labels, preds, 
            f'{workdir}/{split}_post.csv')


    return best_state
    


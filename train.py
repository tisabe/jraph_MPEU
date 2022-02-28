"""Main training loop/update function."""

import os
from typing import (
    NamedTuple, 
    Callable, 
    Dict, 
    Iterable, 
    Sequence, 
    Tuple,
    Optional,
    Any,
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
import functools
import pickle

# import custom functions
from models import GNN
from utils import *
from input_pipeline import get_datasets
from input_pipeline import DataReader



class Updater:
  """A stateless abstraction around an init_fn/update_fn pair.
  This extracts some common boilerplate from the training loop.
  """

  def __init__(self, net_init, loss_fn,
               optimizer: optax.GradientTransformation):
    self._net_init = net_init
    self._loss_fn = loss_fn
    self._opt = optimizer

  @functools.partial(jax.jit, static_argnums=0)
  def init(self, rng, data):
    """Initializes state of the updater."""
    out_rng, init_rng = jax.random.split(rng)
    params = self._net_init(init_rng, data)
    opt_state = self._opt.init(params)
    out = dict(
        step=np.array(0),
        rng=out_rng,
        opt_state=opt_state,
        params=params,
    )
    return out

  @functools.partial(jax.jit, static_argnums=0)
  def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
    """Updates the state using some data and returns metrics."""
    rng, new_rng = jax.random.split(state['rng'])
    params = state['params']
    loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

    updates, opt_state = self._opt.update(g, state['opt_state'])
    params = optax.apply_updates(params, updates)

    new_state = {
        'step': state['step'] + 1,
        'rng': new_rng,
        'opt_state': opt_state,
        'params': params,
    }

    metrics = {
        'step': state['step'],
        'loss': loss,
    }
    return new_state, metrics


class CheckpointingUpdater:
    """A didactic checkpointing wrapper around an Updater.
    A more mature checkpointing implementation might:
    - Use np.savez() to store the core data instead of pickle.
    - Not block JAX async dispatch.
    - Automatically garbage collect old checkpoints.
    """
    def __init__(self,
                inner: Updater,
                checkpoint_dir: str,
                checkpoint_every_n: int = 10):
        self._inner = inner
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n

    def _checkpoint_paths(self):
        return [p for p in os.listdir(self._checkpoint_dir) if 'checkpoint_' in p]

    def init(self, rng, data):
        """Initialize experiment state."""
        if not os.path.exists(self._checkpoint_dir) or not self._checkpoint_paths():
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            return self._inner.init(rng, data)
        else:
            checkpoint = os.path.join(self._checkpoint_dir,
                                max(self._checkpoint_paths()))
            logging.info('Loading checkpoint from %s', checkpoint)
        with open(checkpoint, 'rb') as f:
            state = pickle.load(f)
            return state

    def update(self, state, data):
        """Update experiment state."""
        # NOTE: This blocks until `state` is computed. If you want to use JAX async
        # dispatch, maintain state['step'] as a NumPy scalar instead of a JAX array.
        # Context: https://jax.readthedocs.io/en/latest/async_dispatch.html
        step = np.array(state['step'])
        if step % self._checkpoint_every_n == 0:
            path = os.path.join(self._checkpoint_dir,
                                'checkpoint_{:07d}.pkl'.format(step))
            checkpoint_state = jax.device_get(state)
            logging.info('Serializing experiment state to %s', path)
            with open(path, 'wb') as f:
                pickle.dump(checkpoint_state, f)

        state, out = self._inner.update(state, data)
        return state, out


def make_result_csv(x, y, path):
    '''Print predictions x versus labels y in a csv at path.'''
    dict_res = {'x': np.array(x).flatten(), 'y': np.array(y).flatten()}
    df = pandas.DataFrame(data=dict_res)
    df.to_csv(path)


def get_globals(graphs: Sequence[jraph.GraphsTuple]) -> Sequence[float]:
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
    """Create an Optax optimizer object."""
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

def get_diff_fn(config: ml_collections.ConfigDict) -> Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[float, float]]:
    """
    OBSOLETE!

    Return difference function depending on str argument in config.loss_type.
    The difference function defines how loss is calculated from label and prediction."""
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
    state: train_state.TrainState, graphs: jraph.GraphsTuple) -> Tuple[
        train_state.TrainState, float]:
    """Perform one update step over the batch of graphs.

    Returns a new TrainState and the loss MAE over the batch.
    """

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
        init_graphs: jraph.GraphsTuple,
        workdir: str) -> train_state.TrainState:
    """Initialize a TrainState object using hyperparameters in config,
    and the init_graphs. This is a representative batch of graphs."""
    # Initialize rng.
    rng = jax.random.PRNGKey(config.seed)

    # Create and initialize network.
    logging.info('Initializing network.')
    rng, init_rng = jax.random.split(rng)
    init_graphs = replace_globals(init_graphs) # initialize globals in graph to zero
    
    net_fn = create_model(config)
    net = hk.without_apply_rng(hk.transform(net_fn))
    # TODO: check changing initializer

    # params = net.init(init_rng, init_graphs) # create weights etc. for the model
    
    # Create the optimizer
    # opt_state = create_optimizer(config)
    # opt_state = optax.GradientTransformation
    # TODO: Change these values.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.5),
        optax.adam(0.001, b1=0.9, b2=0.99))


    def loss_fn(params, graphs, net):
        # curr_state = state.replace(params=params)

        labels = graphs.globals
        graphs = replace_globals(graphs)

        mask = get_valid_mask(labels, graphs)
        pred_graphs = net.apply(params, graphs)
        predictions = pred_graphs.globals
        labels = jnp.expand_dims(labels, 1)
        sq_diff = jnp.square((predictions - labels)*mask)
        # TODO: make different loss functions available in config
        loss = jnp.sum(sq_diff)
        mean_loss = loss / jnp.sum(mask)

        return mean_loss, (loss)

    updater = Updater(net.init, loss_fn, optimizer)
    updater = CheckpointingUpdater(
        updater, checkpoint_dir=os.path.join(workdir, 'checkpoints'))
    rng = jax.random.PRNGKey(428)
    data = init_graphs  # Might not work.
    state = updater.init(rng, data)

    # logging.info(f'Init_lr: {config.init_lr}')

    # Create the training state
    # state = train_state.TrainState.create(
    #     apply_fn=net.apply, params=params, tx=opt_state)

    return updater, state


def restore_checkpoint(state, checkpoint_dir):
    """Get param values/step # and the optimizer state from model checkpoint."""
    with open((checkpoint_dir+'/params.pickle'), 'rb') as handle:
        state_dict = pickle.load(handle)
    return state.replace(
        params=state_dict['params'],
        step=state_dict['step'],
        tx=state_dict['opt_state'])
    

def save_checkpoint(state, checkpoint_dir):
    """Save the params/step/optimizer state as a pickle file."""
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    state_dict = {
        'params': state.params,
        'step': state.step,
        'opt_state': state.tx}
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


# def train(
#     config: ml_collections.ConfigDict,
#     datasets: Dict[str, Sequence[jraph.GraphsTuple]],
#     workdir: Optional[str] = None
# ) -> Tuple[train_state.TrainState, float]:
#     '''Train a model using training data in dataset and validation data 
#     in datasets for early stopping and model selection.
    
#     The globals of training and validation graphs need to be normalized,
#     and the loss will be on normalized errors.'''
    
#     reader_train = DataReader(datasets['train'], config.batch_size, 
#         repeat=True, seed=config.seed)
#     init_graphs = next(reader_train)

#     state = init_state(config, init_graphs)
    
#     if workdir is not None:
#         # Set up checkpointing of the model.
#         checkpoint_dir = os.path.join(workdir, 'checkpoints')
#     # start at step 1 (or state.step + 1 if state was restored)
#     initial_step = int(state.step) + 1
#     # TODO: get some framework for automatic checkpoint restoring

#     loss_queue = []
#     params_queue = []
#     best_params = None
    
#     logging.info('Starting training.')
#     for step in range(initial_step, config.num_train_steps_max + 1):
#         # Perform a training step
#         graphs = next(reader_train)
#         state, loss = train_step(state, graphs)

#         is_last_step = (step == config.num_train_steps_max - 1)
#         # evaluate model on train, test and validation data
#         if step % config.eval_every_steps == 0 or is_last_step:
#             eval_loss = evaluate_split(state, datasets['validation'],
#                 config.batch_size)
#             logging.info(f'validation MSE: {eval_loss}')
            
#             loss_queue.append(eval_loss)
#             params_queue.append(state.params)
#             # only test for early stopping after the first interval
#             if step > config.early_stopping_steps:
#                 # stop if new loss higher than loss at beginning of interval
#                 if eval_loss > loss_queue[0]:
#                     break
#                 else:
#                     # otherwise delete the element at beginning of queue
#                     loss_queue.pop(0)
#                     params_queue.pop(0)

#     # save parameters of best model
#     index = np.argmin(loss_queue)
#     # get lowest validation loss
#     min_loss = loss_queue[index]
#     params = params_queue[index]
#     if workdir is not None:
#         with open((workdir+'/params.pickle'), 'wb') as handle:
#             pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     # save predictions of the best model
#     best_state = state.replace(params=params) # restore the state with best params
    
#     return best_state, min_loss

    
def train_and_evaluate(
        config: ml_collections.ConfigDict,
        workdir: str) -> train_state.TrainState:
    
    logging.info('Loading datasets.')
    datasets, datasets_raw, mean, std = get_datasets(config)
    logging.info(f'Number of node classes: {config.max_atomic_number}')

    init_graphs = next(datasets['train'])
    init_graphs = replace_globals(init_graphs) # initialize globals in graph to zero
    
    # Create the training state
    updater, state = init_state(config, init_graphs)
    
    # # Set up checkpointing of the model.
    # checkpoint_dir = os.path.join(workdir, 'checkpoints')

    # set up saving of losses
    splits = ['train', 'validation', 'test']
    loss_dict = {}
    for split in splits:
        loss_dict[split] = []

    # TODO: need to check if saving acutally works, change this code
    # to use checkpoint_updater class.
    # if config.restore:
    #     state = restore_checkpoint(state, checkpoint_dir)
    #     loss_dict = restore_loss_curve(checkpoint_dir, splits, std)

    # TODO: No longer needed since Updater.init creates step=0 in state dict.
    # # start at step 1 (or state.step + 1 if state was restored)
    # initial_step = int(state.step) + 1
    
    # # Make a loss queue to compare with earlier losses
    # loss_queue = []
    # params_queue = []
    # best_params = None

    # # Begin training loop.
    # logging.info('Starting training.')
    # time_logger = Time_logger(config)

    # TODO: Should step start at 0?
    for step in range(0, config.num_train_steps_max + 1):
        # Perform a training step
        graphs = next(datasets['train'])
        state, loss = updater.update(state, graphs)
        # state, loss = train_step(state, graphs)

        # # Log periodically
        # is_last_step = (step == config.num_train_steps_max)
        # if step % config.log_every_steps == 0:
        #     time_logger.log_eta(step)
        
        # # evaluate model on train, test and validation data
        # if step % config.eval_every_steps == 0 or is_last_step:
        #     eval_loss = evaluate_model(state, datasets, splits)
        #     for split in splits:
        #         logging.info(f'MSE {split}: {eval_loss[split]}')
        #         loss_dict[split].append([step, eval_loss[split]])
            
        #     loss_queue.append(eval_loss['validation'])
        #     params_queue.append(state.params)
        #     # only test for early stopping after the first interval
        #     if step > config.early_stopping_steps:
        #         # stop if new loss higher than loss at beginning of interval
        #         if eval_loss['validation'] > loss_queue[0]:
        #             logging.info('Stopping early.')
        #             break
        #         else:
        #             # otherwise delete the element at beginning of queue
        #             loss_queue.pop(0)
        #             params_queue.pop(0)
        
        # # Checkpoint model, if required
        # if step % config.checkpoint_every_steps == 0 or is_last_step:
        #     save_checkpoint(state, checkpoint_dir)
        #     # save the loss curves
        #     save_loss_curve(loss_dict, checkpoint_dir, splits, std)
        # if step==config.num_train_steps_max:
        #     logging.info('Reached maximum number of steps without early stopping.')

    # # save parameters of best model
    # index = np.argmin(loss_queue)
    # params = params_queue[index]
    # with open((workdir+'/params.pickle'), 'wb') as handle:
    #     pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # save predictions of the best model
    # best_state = state.replace(params=params) # restore the state with best params
    # #pred_dict = predict(best_state, datasets_raw, splits)
    
    # for split in splits:
    #     loss_split = np.array(loss_dict[split])
    #     # convert loss column to eV
    #     loss_split[:,1] = loss_split[:,1]*std
    #     # save the loss curves
    #     np.savetxt(f'{workdir}/{split}_loss.csv', 
    #         np.array(loss_split), delimiter=",")

    #     # save the predictions and labels
    #     #labels = get_globals(datasets[split].data)
    #     labels = get_globals(datasets_raw[split])
    #     preds = predict_split(best_state, datasets_raw[split], config)
    #     preds = scale_targets_config(datasets_raw[split], preds, mean, std, config)
    #     #preds = get_labels_original(datasets_raw[split], preds, config.label_str)
    #     labels = np.array(labels)
    #     preds = np.array(preds)
    #     make_result_csv(
    #         labels, preds, 
    #         f'{workdir}/{split}_post.csv')


    return best_state
    


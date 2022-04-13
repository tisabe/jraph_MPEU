"""Main training loop/update function."""

import os
import functools
import pickle
from typing import (
    Callable,
    Dict,
    Iterable,
    Sequence,
    Tuple,
    Any,
    Mapping
)

from absl import logging
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import optax
import haiku as hk
import pandas as pd

# import custom functions
from models import GNN
from utils import (
    #Time_logger,
    replace_globals,
    get_valid_mask,
    save_config
)
from input_pipeline import (
    get_datasets,
    DataReader,
)



class Updater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """

    def __init__(self, net, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net.init
        self._net_apply = net.apply
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        state = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return state

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: jraph.GraphsTuple):
        """Updates the state using some data and returns metrics."""
        #rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        (loss, _), grad = jax.value_and_grad(
            self._loss_fn, has_aux=True)(params, data, self._net_apply)

        updates, opt_state = self._opt.update(grad, state['opt_state'])
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': 0,
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
                 checkpoint_every_n: int):
        self._inner = inner
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n

    def _checkpoint_paths(self):
        return [p for p in os.listdir(self._checkpoint_dir) if 'checkpoint_' in p]

    def init(self, rng, data):
        """Initialize experiment state."""
        # TODO: include argument to ignore previous checkpoints
        if not os.path.exists(self._checkpoint_dir) or not self._checkpoint_paths():
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            return self._inner.init(rng, data)
        else:
            checkpoint = os.path.join(
                self._checkpoint_dir,
                max(self._checkpoint_paths()))
            logging.info('Loading checkpoint from %s', checkpoint)
        with open(checkpoint, 'rb') as state_file:
            state = pickle.load(state_file)
            return state

    def update(self, state, data):
        """Update experiment state."""
        # NOTE: This blocks until `state` is computed. If you want to use JAX async
        # dispatch, maintain state['step'] as a NumPy scalar instead of a JAX array.
        # Context: https://jax.readthedocs.io/en/latest/async_dispatch.html
        state, metrics = self._inner.update(state, data)

        step = np.array(state['step'])
        if step % self._checkpoint_every_n == 0:
            path = os.path.join(self._checkpoint_dir,
                                'checkpoint_{:07d}.pkl'.format(step))
            checkpoint_state = jax.device_get(state)
            logging.info('Serializing experiment state to %s', path)
            with open(path, 'wb') as state_file:
                pickle.dump(checkpoint_state, state_file)

        return state, metrics


class Evaluater:
    """A class to evaluate the model with, save and checkpoint loss metrics.

    Args:
        net: network function, made by haiku.Transform, has a .apply function
        loss_fn: callable that computes loss using model parameters, input graph
            and a function that applies the network to the graph
        checkpoint_dir: directory to store checkpoints, metrics and best state in
        checkpoint_every_n: after how many steps a new checkpoint should be saved
        eval_every_n: after how many steps the model should be evaluated

    Intended use case:
        After initializing the evaluater with:

        evaluater = Evaluater(net, loss_fn,
            os.path.join(workdir, 'checkpoints'),
            config.checkpoint_every_steps,
            config.eval_every_steps)

        and once the splits to evaluate on are known, calling:

        evaluater.init_loss_lists(eval_splits)

        In the main training loop only the update function needs to be called:

        early_stop = evaluater.update(state, datasets, eval_splits)
    """
    def __init__(
            self, net, loss_fn, checkpoint_dir: str,
            checkpoint_every_n: int, eval_every_n: int):
        self._net_init = net.init
        self._net_apply = net.apply
        self._loss_fn = loss_fn
        self.val_queue = []
        self.loss_dict = {}
        self.best_state = None # save the state with lowest validation error in best state
        self.lowest_val_loss = None
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n
        self._eval_every_n = eval_every_n
        # load loss curve if metrics file exists in checkpoint_dir
        metrics_path = os.path.join(self._checkpoint_dir, 'metrics.pkl')
        best_state_path = os.path.join(
            self._checkpoint_dir,
            'best_state.pkl')
        if os.path.exists(metrics_path):
            # load metrics, if they have been saved before
            logging.info('Loading metrics from %s', metrics_path)
            with open(metrics_path, 'rb') as metrics_file:
                self._metrics_dict = pickle.load(metrics_file)
            self._loaded_metrics = True
            # load best state and lowest loss
            with open(best_state_path, 'rb') as state_file:
                best_state_dict = pickle.load(state_file)
                self.best_state = best_state_dict['state']
                self.lowest_val_loss = best_state_dict['loss']
        else:
            self._loaded_metrics = False

    def checkpoint_best_state(self):
        """Save/keep track of the lowest loss and associated state."""
        state_loss_dict = {'state': self.best_state,
                           'loss': self.lowest_val_loss}
        path = os.path.join(self._checkpoint_dir,
                            'best_state.pkl')
        with open(path, 'wb') as state_file:
            pickle.dump(state_loss_dict, state_file)

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_step(
            self, state: dict, graphs: jraph.GraphsTuple) -> float:
        """Calculate the mean loss for a batch of graphs."""
        (mean_loss, (mae)) = self._loss_fn(state['params'], graphs, self._net_apply)
        return [mean_loss, mae]

    def evaluate_split(
            self,
            state: dict,
            graphs: Sequence[jraph.GraphsTuple],
            batch_size: int) -> float:
        """Return mean loss for all graphs in graphs."""

        reader = DataReader(
            data=graphs, batch_size=batch_size, repeat=False)

        loss_list = [self._evaluate_step(state, batch) for batch in reader]
        return np.mean(loss_list, axis=0)

    def evaluate_model(
            self,
            state: Dict,
            datasets: Dict[str, Iterable[jraph.GraphsTuple]],
            splits: Iterable[str]) -> Dict[str, float]:
        """Return mean loss for every split in splits.
        
        Also save a checkpoint of the best state, so it is not lost if loss
        decreases, but training stops before the next checkpoint."""
        loss_dict = {}
        for split in splits:
            loss_dict[split] = self.evaluate_split(
                state, datasets[split].data, datasets[split].batch_size)
            if split == 'validation':
                if self.best_state is None or loss_dict[split][0] < self.lowest_val_loss:
                    self.best_state = state.copy()
                    self.lowest_val_loss = loss_dict[split][0]
                    self.checkpoint_best_state()
        return loss_dict

    def init_loss_lists(self, splits):
        """Initialize a dict to save evaluation losses in."""

        if not self._loaded_metrics:
            for split in splits:
                self.loss_dict[split] = []
            # initialize a queue with validation losses for early stopping
            self.early_stopping_queue = []
        else:
            for split in splits:
                self.loss_dict[split] = self._metrics_dict[split]
            self.early_stopping_queue = self._metrics_dict['queue']

    def save_losses(self, loss_dict, splits, step):
        """Append values in loss_dict to self.loss_dict for all splits.

        Also create the local early stopping queue.
        """
        for split in splits:
            self.loss_dict[split].append([step, loss_dict[split]])
            if split == 'validation':
                self.early_stopping_queue.append(loss_dict[split][0])

    def check_early_stopping(self):
        """Check the early stopping criterion.

        If the newest validaiton loss in self.early_stopping_queue is higher
        than the zeroth one return True for early stopping. Otherwise, delete
        the zeroth element in queue and return False for no early stopping.
        """
        queue = self.early_stopping_queue  # abbreviation
        if queue[-1] > queue[0]:  # check for early stopping condition
            return True
        else:
            queue.pop(0)  # Note: also modifies self.early_stopping_queue
            return False

    def checkpoint_losses(self):
        """Save metrics to a dictionary at checkpoint."""
        metrics_dict = self.loss_dict.copy()
        metrics_dict['queue'] = self.early_stopping_queue
        # save metrics
        path = os.path.join(self._checkpoint_dir, 'metrics.pkl')
        with open(path, 'wb') as metrics_file:
            pickle.dump(metrics_dict, metrics_file)

    def update(self, state, datasets, eval_splits):
        """Does evaluation, checkpointing and checks for early stopping.

        Calculate and save loss metrics, checkpoint model and check for early
        stopping.
        """
        step = state['step']
        if step % self._eval_every_n == 0:
            eval_loss = self.evaluate_model(state, datasets, eval_splits)
            for split in eval_splits:
                logging.info(f'MSE {split}: {eval_loss[split]}')
            self.save_losses(eval_loss, eval_splits, step)
            early_stop = self.check_early_stopping()
        else:
            early_stop = False

        if step % self._checkpoint_every_n == 0:
            self.checkpoint_losses()
            self.checkpoint_best_state()

        return early_stop


def make_result_csv(predictions, labels, path):
    """Print predictions x versus labels y in a csv at path.

    TODO: Want to refactor this function.
    """
    dict_res = {
        'x': np.array(predictions).flatten(),
        'y': np.array(labels).flatten()}
    result_df = pd.DataFrame(data=dict_res)
    result_df.to_csv(path)


def get_globals(graphs: Sequence[jraph.GraphsTuple]) -> Sequence[float]:
    """Return list of global labels for each graph."""
    labels = []
    for graph in graphs:
        labels.append(float(graph.globals))

    return labels


def create_model(config: ml_collections.ConfigDict):
    """Return a function that applies the graph model."""
    return GNN(config)


def cosine_warm_restarts(
        init_value: float,
        decay_steps: int,
) -> Callable[[int], float]:
    '''Return a function that implements a cosine schedule with warm restarts.
    For more details see: https://arxiv.org/abs/1608.03983'''

    if not decay_steps > 0:
        raise ValueError('The cosine_decay_schedule requires positive decay_steps!')

    def schedule(count):
        # TODO: implement multiplier
        count_since_restart = count % decay_steps
        cosine = 0.5 * (1 + jnp.cos(jnp.pi * count_since_restart / decay_steps))
        return init_value * cosine

    return schedule


def create_optimizer(
        config: ml_collections.ConfigDict) -> optax.GradientTransformation:
    """Create an Optax optimizer object."""
    # TODO: consider including gradient clipping
    if config.schedule == 'exponential_decay':
        learning_rate = optax.exponential_decay(
            init_value=config.init_lr,
            transition_steps=config.transition_steps,
            decay_rate=config.decay_rate,
            staircase=True)
    elif config.schedule == 'cosine_decay':
        learning_rate = cosine_warm_restarts(
            init_value=config.init_lr,
            decay_steps=config.transition_steps)
    else:
        raise ValueError(f'Unsupported schedule: {config.schedule}.')

    if config.optimizer == 'adam':
        return optax.adam(learning_rate=learning_rate)
    raise ValueError(f'Unsupported optimizer: {config.optimizer}.')


def predict_split(
        state: Dict,
        dataset_raw: Sequence[jraph.GraphsTuple],
        config: ml_collections.ConfigDict) -> Sequence[float]:
    """Combine predictions into a list that's returned.

    Use the MPNN from the state variable and feed in the dataset_raw graphs.

    TODO: Might want to refactor this so that training is is one file and
    eval is in another file.
    """
    preds = np.array([])
    reader_new = DataReader(
        data=dataset_raw,
        batch_size=config.batch_size, repeat=False)

    for graphs in reader_new:
        labels = graphs.globals
        graphs = replace_globals(graphs)

        mask = get_valid_mask(labels, graphs)
        # Apply the MPNN on the input graphs.
        pred_graphs = state.apply_fn(state.params, graphs)
        preds_batch = pred_graphs.globals
        # Throw away all padding labels
        preds_batch_valid = preds_batch[mask]
        # Update predictions list.
        preds = np.concatenate((preds, preds_batch_valid), axis=0)

    return preds


def init_state(
        config: ml_collections.ConfigDict,
        init_graphs: jraph.GraphsTuple,
        workdir: str) -> Tuple[CheckpointingUpdater, Dict, Evaluater]:
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

    # Create the optimizer
    optimizer = create_optimizer(config)

    def loss_fn(params, graphs, net_apply):
        labels = graphs.globals
        graphs = replace_globals(graphs)

        mask = get_valid_mask(labels, graphs)
        pred_graphs = net_apply(params, graphs)
        predictions = pred_graphs.globals
        labels = jnp.expand_dims(labels, 1)
        sq_diff = jnp.square((predictions - labels)*mask)
        # TODO: make different loss functions available in config
        loss = jnp.sum(sq_diff)
        mean_loss = loss / jnp.sum(mask)
        absolute_error = jnp.sum(jnp.abs((predictions - labels)*mask))
        mae = absolute_error /jnp.sum(mask)

        return mean_loss, mae

    updater = Updater(net, loss_fn, optimizer)
    updater = CheckpointingUpdater(
        updater, os.path.join(workdir, 'checkpoints'),
        config.checkpoint_every_steps)
    evaluater = Evaluater(
        net, loss_fn,
        os.path.join(workdir, 'checkpoints'),
        config.checkpoint_every_steps,
        config.eval_every_steps)

    state = updater.init(init_rng, init_graphs)

    return updater, state, evaluater


def restore_loss_curve(ckpt_dir, splits, std):
    """Load the loss curve as a dictionary.

    TODO: Refactor.
    """
    loss_dict = {}
    for split in splits:
        loss_split = np.loadtxt(
            f'{ckpt_dir}/{split}_loss.csv',
            delimiter=',', ndmin=2)
        loss_split[:, 1] = loss_split[:, 1]/std
        loss_dict[split] = loss_split.tolist()
    return loss_dict


def save_loss_curve(loss_dict, ckpt_dir, splits, std):
    """Save the loss curve from the loss dictionary.

    TODO: Refactor.
    """
    for split in splits:
        loss_split = np.array(loss_dict[split])
        # convert loss column to eV
        loss_split[:, 1] = loss_split[:, 1]*std
        # save the loss curves
        np.savetxt(
            f'{ckpt_dir}/{split}_loss.csv',
            np.array(loss_split), delimiter=',')


def train_and_evaluate(
        config: ml_collections.ConfigDict,
        workdir: str) -> Dict:
    """Train the model and evaluate it."""
    logging.info('Loading datasets.')
    datasets, _, _, _ = get_datasets(config)
    logging.info(f'Number of node classes: {config.max_atomic_number}')

    # save the config in txt for later inspection
    save_config(config, workdir)

    init_graphs = next(datasets['train'])
    # Initialize globals in graph to zero. Don't want to give the model
    # the right answer. The model's not using them now anyway.
    init_graphs = replace_globals(init_graphs)
    # Create the training state.
    updater, state, evaluater = init_state(config, init_graphs, workdir)

    # Decide on splits of data on which to evaluate.
    eval_splits = ['train', 'validation', 'test']
    # Set up saving of losses.
    evaluater.init_loss_lists(eval_splits)

    # Start at step 1 (or state.step + 1 if state was restored).
    # state['step'] is initialized to 0 if no checkpoint was loaded.
    initial_step = int(state['step']) + 1

    # Begin training loop.
    logging.info('Starting training.')
    # time_logger = Time_logger(config)

    for step in range(initial_step, config.num_train_steps_max + 1):
        # Perform a training step. Get next training graphs.
        graphs = next(datasets['train'])
        # Update the weights after a gradient step and report the
        # state/losses/optimizer gradient. The loss returned here is the loss
        # on a batch not on the full training dataset.
        state, loss_metrics = updater.update(state, graphs)

        # Log periodically the losses/step count.
        # TODO: Use the last step to break out of this training loop if
        # we have already completed the max number of training steps.
        is_last_step = (step == config.num_train_steps_max)
        if step % config.log_every_steps == 0:
            # TODO: Add timing metrics.
            #time_logger.log_eta(step)
            logging.info(f'Step {step} train loss: {loss_metrics["loss"]}')

        # Get evaluation on all splits of the data (train/validation/test),
        # checkpoint if needed and
        # check if we should be stopping early.
        early_stop = evaluater.update(state, datasets, eval_splits)

        if early_stop:
            logging.info(f'Loss converged at step {step}, stopping early.')
            break

        # No need to break if it's the last step since the loop terminates
        # automatically when reaching the last step.
        if is_last_step:
            logging.info(
                'Reached maximum number of steps without early stopping.')

    lowest_val_loss = evaluater.lowest_val_loss
    logging.info(f'Lowest validation loss: {lowest_val_loss}')

    return evaluater, lowest_val_loss

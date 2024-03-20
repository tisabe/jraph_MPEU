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

# import custom functions
from jraph_MPEU.models.loading import create_model
from jraph_MPEU.utils import (
    replace_globals,
    get_valid_mask,
    save_config
)
from jraph_MPEU.input_pipeline import (
    get_datasets,
    DataReader,
)

# maximum loss, if training batch loss exceeds this, stop training
_MAX_TRAIN_LOSS = 1e10

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
        params, hk_state = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        state = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
            hk_state=hk_state
        )
        return state

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: jraph.GraphsTuple):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        hk_state = state['hk_state']
        (loss, (_, hk_state)), grad = jax.value_and_grad(
            self._loss_fn, has_aux=True)(params, hk_state, rng, data, self._net_apply)

        updates, opt_state = self._opt.update(grad, state['opt_state'], params)
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
            'hk_state': hk_state
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
                 checkpoint_every_n: int,
                 num_checkpoints: int):
        self._inner = inner
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n
        self._num_checkpoints = num_checkpoints

    def _checkpoint_paths(self):
        return [p for p in os.listdir(self._checkpoint_dir) if 'checkpoint_' in p]

    def init(self, rng, data):
        """Initialize experiment state."""
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
        # NOTE: This blocks until `state` is computed. If you want to use JAX
        # async dispatch, maintain state['step'] as a NumPy scalar instead of a
        # JAX array.
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

            # check if too many checkpoints are currently stored
            paths = self._checkpoint_paths()
            # if there are too many checkpoints, remove the oldest
            if len(paths) > self._num_checkpoints:
                os.remove(
                    os.path.join(self._checkpoint_dir, min(paths))
                )

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
            checkpoint_every_n: int, eval_every_n: int,
            early_stopping_steps: int, batch_size: int,
            metric_names: str = 'RMSE/MAE'):
        self._net_apply = net.apply
        self._loss_fn = loss_fn
        self.early_stopping_queue = []
        self.loss_dict = {}
        self.rng = None  # initialize rng for later assign in evaluate model
        self.best_state = None # save the state with lowest validation error in best state
        self.lowest_val_loss = None
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n
        self._early_stopping_steps = early_stopping_steps
        self._eval_every_n = eval_every_n
        self._metric_names = metric_names
        self._loss_scalar = 1.0
        self._batch_size = batch_size
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

    def set_loss_scalar(self, scalar):
        """Set a scalar to multiply the saved losses with.

        This scalar will be multiplied with both MSE loss and MAE, the intended
        usecase is to use the standard deviation of the dataset, if the dataset
        has been normalized, to scale the saved losses back to the original
        value.
        """
        self._loss_scalar = scalar

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
        """Calculate the mean loss for a batch of graphs. Returns scaled MSE
        and MAE over batch"""
        state['rng'], new_rng = jax.random.split(state['rng'])
        (mean_loss, (mae, _)) = self._loss_fn(
            state['params'], state['hk_state'], new_rng, graphs, self._net_apply)
        return [mean_loss, mae]

    def evaluate_split(
            self,
            state: dict,
            graphs: Sequence[jraph.GraphsTuple],
            batch_size: int) -> float:
        """Return mean loss for all graphs in graphs. First return value is
        RMSE, second value is MAE, both scaled back using std of dataset."""

        reader = DataReader(
            data=graphs, batch_size=batch_size, repeat=False, seed=0)

        loss_list = []
        weights_list = []
        for batch in reader:
            # get number of graphs in batch as weight for this batch
            weights_list.append(batch_size - jraph.get_number_of_padding_with_graphs_graphs(batch))
            loss_list.append(self._evaluate_step(state, batch))
        averaged = np.average(loss_list, axis=0, weights=weights_list)
        return self._loss_scalar*np.array([np.sqrt(averaged[0]), averaged[1]])

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
                state, datasets[split], self._batch_size)
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

    def check_early_stopping(self, step):
        """Check the early stopping criterion.

        If the newest validation loss in self.early_stopping_queue is higher
        than the zeroth one return True for early stopping. Otherwise, delete
        the zeroth element in queue and return False for no early stopping.
        """
        queue = self.early_stopping_queue  # abbreviation
        if step > self._early_stopping_steps:
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
                logging.info(f'{self._metric_names} {split}: {eval_loss[split]}')
            self.save_losses(eval_loss, eval_splits, step)
            early_stop = self.check_early_stopping(step)
        else:
            early_stop = False

        if step % self._checkpoint_every_n == 0:
            self.checkpoint_losses()
            self.checkpoint_best_state()

        return early_stop


def get_globals(graphs: Sequence[jraph.GraphsTuple]) -> Sequence[float]:
    """Return list of global labels for each graph."""
    labels = []
    for graph in graphs:
        labels.append(float(graph.globals))

    return labels


def cosine_warm_restarts(
        init_value: float,
        decay_steps: int,
) -> Callable[[int], float]:
    """Return a function that implements a cosine schedule with warm restarts.
    For more details see: https://arxiv.org/abs/1608.03983"""

    if not decay_steps > 0:
        raise ValueError('The cosine_decay_schedule requires positive decay_steps!')

    def schedule(count):
        count_since_restart = count % decay_steps
        cosine = 0.5 * (1 + jnp.cos(jnp.pi * count_since_restart / decay_steps))
        return init_value * cosine

    return schedule


def create_optimizer(
        config: ml_collections.ConfigDict) -> optax.GradientTransformation:
    """Create an Optax optimizer object."""
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
    elif config.optimizer == 'adamw':
        return optax.adamw(
            learning_rate=learning_rate, weight_decay=config.weight_decay)

    raise ValueError(f'Unsupported optimizer: {config.optimizer}.')


def loss_fn_mse(params, state, rng, graphs, net_apply):
    """Mean squared error loss function for regression."""
    labels = graphs.globals
    graphs = replace_globals(graphs)

    mask = get_valid_mask(graphs)
    pred_graphs, new_state = net_apply(params, state, rng, graphs)
    predictions = pred_graphs.globals
    labels = jnp.expand_dims(labels, 1)
    sq_diff = jnp.square((predictions - labels)*mask)

    loss = jnp.sum(sq_diff)
    mean_loss = loss / jnp.sum(mask)
    absolute_error = jnp.sum(jnp.abs((predictions - labels)*mask))
    mae = absolute_error /jnp.sum(mask)

    return mean_loss, (mae, new_state)


def loss_fn_bce(params, state, rng, graphs, net_apply):
    """Binary cross entropy loss function for classification."""
    labels = graphs.globals
    graphs = replace_globals(graphs)
    targets = jax.nn.one_hot(labels, 2)

    # try get_valid_mask function instead
    mask = jraph.get_graph_padding_mask(graphs)
    pred_graphs, new_state = net_apply(params, state, rng, graphs)
    # compute class probabilities
    preds = jax.nn.log_softmax(pred_graphs.globals)
    # Cross entropy loss, note: we average only over valid (unmasked) graphs
    loss = -jnp.sum(preds * targets * mask[:, None])/jnp.sum(mask)

    # Accuracy taking into account the mask.
    accuracy = jnp.sum(
        (jnp.argmax(pred_graphs.globals, axis=1) == labels) * mask)/jnp.sum(mask)
    return loss, (accuracy, new_state)


def init_state(
        config: ml_collections.ConfigDict,
        init_graphs: jraph.GraphsTuple,
        workdir: str) -> Tuple[CheckpointingUpdater, Dict, Evaluater]:
    """Initialize a TrainState object using hyperparameters in config,
    and the init_graphs. This is a representative batch of graphs."""
    # Initialize rng.
    rng = jax.random.PRNGKey(config.seed_weights)

    # Create and initialize network.
    logging.info('Initializing network.')
    rng, init_rng = jax.random.split(rng)
    init_graphs = replace_globals(init_graphs) # initialize globals in graph to zero

    net_fn_eval = create_model(config, is_training=False)
    net_eval = hk.transform_with_state(net_fn_eval)

    net_fn_train = create_model(config, is_training=True)
    net_train = hk.transform_with_state(net_fn_train)

    # Create the optimizer
    optimizer = create_optimizer(config)

    # determine which loss function to use
    if config.label_type == 'scalar':
        loss_fn = loss_fn_mse
        metric_names = 'RMSE/MAE'
    else:
        loss_fn = loss_fn_bce
        metric_names = 'BCE/Acc.'

    updater = Updater(net_train, loss_fn, optimizer)
    updater = CheckpointingUpdater(
        updater, os.path.join(workdir, 'checkpoints'),
        config.checkpoint_every_steps,
        config.num_checkpoints)

    evaluater = Evaluater(
        net_eval, loss_fn,
        os.path.join(workdir, 'checkpoints'),
        config.checkpoint_every_steps,
        config.eval_every_steps,
        config.early_stopping_steps,
        config.batch_size,
        metric_names)

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
    datasets, norm_dict = get_datasets(config, workdir)
    logging.info(f'Number of node classes: {config.max_atomic_number}')

    # save the config in txt for later inspection
    save_config(config, workdir)

    # initialize data reader with training data
    train_reader = DataReader(
        data=datasets['train'],
        batch_size=config.batch_size,
        repeat=True,
        seed=config.seed_datareader)

    init_graphs = next(train_reader)
    # Initialize globals in graph to zero. Don't want to give the model
    # the right answer. The model's not using them now anyway.
    init_graphs = replace_globals(init_graphs)
    # Create the training state.
    updater, state, evaluater = init_state(config, init_graphs, workdir)

    # calculate and print parameter size
    params = state['params']
    num_params = hk.data_structures.tree_size(params)
    byte_size = hk.data_structures.tree_bytes(params)
    logging.info(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')

    # Decide on splits of data on which to evaluate.
    eval_splits = ['train', 'validation', 'test']
    # Set up saving of losses.
    evaluater.init_loss_lists(eval_splits)
    if config.label_type == 'scalar':
        evaluater.set_loss_scalar(norm_dict['std'])
    else:
        evaluater.set_loss_scalar(1.0)

    # Start at step 1 (or state.step + 1 if state was restored).
    # state['step'] is initialized to 0 if no checkpoint was loaded.
    initial_step = int(state['step']) + 1

    # Begin training loop.
    logging.info('Starting training.')

    for step in range(initial_step, config.num_train_steps_max + 1):
        # Perform a training step. Get next training graphs.
        graphs = next(train_reader)
        # Update the weights after a gradient step and report the
        # state/losses/optimizer gradient. The loss returned here is the loss
        # on a batch not on the full training dataset.
        state, loss_metrics = updater.update(state, graphs)

        # Log periodically the losses/step count.
        is_last_step = step == config.num_train_steps_max
        if step % config.log_every_steps == 0:
            logging.info(f'Step {step} train loss: {loss_metrics["loss"]}')

        # catch a NaN or too high loss, stop training if it happens
        if (np.isnan(loss_metrics["loss"]) or
                (loss_metrics["loss"] > _MAX_TRAIN_LOSS)):
            logging.info('Invalid loss, stopping early.')
            # create a file that signals that training stopped early
            if not os.path.exists(workdir + '/ABORTED_EARLY'):
                with open(workdir + '/ABORTED_EARLY', 'w', encoding="utf-8"):
                    pass
            break

        # Get evaluation on all splits of the data (train/validation/test),
        # checkpoint if needed and
        # check if we should be stopping early.
        early_stop = evaluater.update(state, datasets, eval_splits)

        if early_stop:
            logging.info(f'Loss converged at step {step}, stopping early.')
            # create a file that signals that training stopped early
            if not os.path.exists(workdir + '/STOPPED_EARLY'):
                with open(workdir + '/STOPPED_EARLY', 'w', encoding="utf-8"):
                    pass
            break

        # No need to break if it's the last step since the loop terminates
        # automatically when reaching the last step.
        if is_last_step:
            logging.info(
                'Reached maximum number of steps without early stopping.')
            if not os.path.exists(workdir + '/REACHED_MAX_STEPS'):
                with open(workdir + '/REACHED_MAX_STEPS', 'w', encoding="utf-8"):
                    pass

    lowest_val_loss = evaluater.lowest_val_loss
    logging.info(f'Lowest validation loss: {lowest_val_loss}')
    return evaluater, lowest_val_loss

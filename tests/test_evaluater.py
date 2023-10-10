"""Test the class and methods of evaluater in train.py. For this, artificial
datasets are produced for which the result is known."""

import unittest
import tempfile
import os

import numpy as np
import jax
import jax.numpy as jnp
import jraph
import haiku as hk

import ml_collections

from jraph_MPEU.utils import replace_globals, get_valid_mask
from jraph_MPEU.input_pipeline import get_datasets, DataReader
from jraph_MPEU_configs import default_test as cfg
from jraph_MPEU.train import Evaluater, create_model


def get_random_graph(n_node):
    """Return a random graphsTuple with n_node nodes, using a jax random key."""
    graph = jraph.GraphsTuple(
        n_node=np.asarray([n_node]), n_edge=np.asarray([n_node]),
        nodes=np.ones((n_node, 4)), edges=np.ones((n_node, 5)),
        globals=np.ones((1, 6)),
        senders=np.arange(n_node), receivers=np.arange(n_node)
    )
    return graph


class UnitTests(unittest.TestCase):
    """Unit and integration test functions in models.py."""
    def test_known_eval(self):
        """Test the evaluater function evaluate_model for a artificial dataset
        with known output.
        """
        config = cfg.get_config()

        with tempfile.TemporaryDirectory() as test_dir:
            datasets, mean, std = get_datasets(config, test_dir)

            init_graphs = datasets['train'][0]

            rng = jax.random.PRNGKey(42)
            rng, init_rng = jax.random.split(rng)

            net_fn_eval = create_model(config, is_training=False)
            net_eval = hk.transform_with_state(net_fn_eval)

            params = net_eval.init(init_rng, init_graphs)

            def loss_fn(params, state, rng, graphs, net_apply):
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

            evaluater = Evaluater(
                net_eval, loss_fn,
                os.path.join(test_dir, 'checkpoints'),
                config.checkpoint_every_steps,
                config.eval_every_steps,
                config.batch_size,
                config.early_stopping_steps)

            eval_splits = ['train', 'validation', 'test']
            evaluater.init_loss_lists(eval_splits)
            evaluater.set_loss_scalar(std)

            loss_dict_expected = {}
            for split in eval_splits:
                graph_reader = DataReader(
                    data=datasets[split],
                    batch_size=config.batch_size, repeat=False)
                # TODO: finish test


if __name__ == '__main__':
    unittest.main()

"""Test the functions in the train module.

Most tests consist of running model training for some steps and then looking at
the resulting metrics and checkpoints.
"""

import tempfile
import unittest
import os

import numpy as np
import jax
from absl import logging
import jax.numpy as jnp
import jraph
import haiku as hk

from jraph_MPEU import train
from jraph_MPEU.input_pipeline import get_datasets
from jraph_MPEU_configs import test_numerics as cfg_num
from jraph_MPEU_configs import default_test as cfg

class TestTrain(unittest.TestCase):
    """Test train.py methods and classes."""
    def setUp(self):
        logging.set_verbosity(logging.WARNING)
        self.config = cfg.get_config()
        self.config.limit_data = 100
        self.assertEqual(self.config.batch_size, 32)
        # get testing datasets
        with tempfile.TemporaryDirectory() as test_dir:
            datasets, _ = get_datasets(self.config, test_dir)
            self.datasets = datasets

    def test_nll(self):
        """Test the negative log likelihood loss for uncertainty quantification."""
        n_node = jnp.array([1, 1, 1, 1, 1, 1])
        n_edge = jnp.array([1, 1, 1, 1, 1, 1])
        #target = jnp.array([0, 0, 0, 0, 0, 0])
        target = jnp.zeros((6))

        graphs = jraph.GraphsTuple(
            nodes=None,
            edges=None,
            n_node=n_node,
            n_edge=n_edge,
            receivers=jnp.array([0, 1, 2, 3, 4, 5]),
            senders=jnp.array([0, 1, 2, 3, 4, 5]),
            globals=target
        )
        # pad the graphs with two additional one-node, one-edge graphs
        graphs_padded = jraph.pad_with_graphs(graphs, 8, 8, 8)
        mu_pred = jnp.ones((6,1))
        sigma_pred = jnp.ones((6,1))*2.

        mse_expected = 1/6*np.sum(np.square(jnp.expand_dims(target, 1)-mu_pred))
        nll_expected = 1/6*np.sum(np.square(
            jnp.expand_dims(target, 1)-mu_pred)/sigma_pred + np.log(sigma_pred))
        mae_expected = np.mean(abs(jnp.expand_dims(target, 1) - mu_pred))

        def net_apply(params, state, rng, graph):
            """Dummy function that changes the global to fixed vector"""
            del params, rng
            graph = jraph.GraphsTuple(
                nodes=None,
                edges=None,
                n_node=n_node,
                n_edge=n_edge,
                receivers=jnp.array([0, 1, 2, 3, 4, 5]),
                senders=jnp.array([0, 1, 2, 3, 4, 5]),
                globals={
                    'mu': mu_pred,
                    'sigma': sigma_pred}
            )
            graph = jraph.pad_with_graphs(graph, 8, 8, 8)
            return graph, state
        params = {}
        rng = 1
        interpolation_steps = 1_000_000
        # At step 0 and step interpolation_steps, the loss should be equivalent to MSE.
        # Since the target are always 0, and we predict mu 1, the MSE is exactly 6.
        state = {'step': 0, 'hk_state': None}
        mean_loss, (mae, new_state) = train.loss_fn_nll(
            params, state, rng, graphs_padded, net_apply)
        self.assertEqual(mean_loss, mse_expected)
        self.assertEqual(mae, mae_expected)
        self.assertEqual(state, new_state)

        state['step'] = interpolation_steps
        mean_loss, (mae, _) = train.loss_fn_nll(
            params, state, rng, graphs_padded, net_apply)
        self.assertEqual(mean_loss, mse_expected)
        self.assertEqual(mae, mae_expected)

        # At step 1.5*interpolation_steps, the loss should be halway between
        # the nll loss and MSE
        state['step'] = int(interpolation_steps*1.5)
        mean_loss, (mae, _) = train.loss_fn_nll(
            params, state, rng, graphs_padded, net_apply)
        self.assertEqual(mean_loss, (mse_expected+nll_expected)/2)
        self.assertEqual(mae, mae_expected)

        # At step 2*interpolation_steps, the loss should be just the nll
        state['step'] = int(interpolation_steps*2)
        mean_loss, (mae, new_state) = train.loss_fn_nll(
            params, state, rng, graphs_padded, net_apply)
        self.assertEqual(mean_loss, nll_expected)
        self.assertEqual(mae, mae_expected)

    def test_cross_entropy(self):
        """Test the binary cross entropy loss function."""
        # create a dummy graph batch
        n_node = jnp.array([1, 1, 1, 1, 1, 1])
        n_edge = jnp.array([1, 1, 1, 1, 1, 1])

        graphs = jraph.GraphsTuple(
            nodes=None,
            edges=None,
            n_node=n_node,
            n_edge=n_edge,
            receivers=jnp.array([0, 1, 2, 3, 4, 5]),
            senders=jnp.array([0, 1, 2, 3, 4, 5]),
            globals=jnp.array([0, 1, 1, 0, 1, 1])
        )
        # pad the graphs with two additional one-node, one-edge graphs
        graphs_padded = jraph.pad_with_graphs(graphs, 8, 8, 8)

        def net_apply(params, state, rng, graph):
            """Dummy function that changes the global to fixed vector"""
            graph = jraph.GraphsTuple(
                nodes=None,
                edges=None,
                n_node=n_node,
                n_edge=n_edge,
                receivers=jnp.array([0, 1, 2, 3, 4, 5]),
                senders=jnp.array([0, 1, 2, 3, 4, 5]),
                globals=jnp.array(
                    [[1, 0.1], [0.1, 1], [1./2, 1./3], [0.1, 1],
                     [0.1, 1], [1./3, 1./2], [3.0, 2.0], [3.0, 2.0]]
                )
            )
            return graph, state
        # test parameters
        params = {}
        state = {'hk_state': None}
        rng = 1

        loss, (acc, new_state) = train.loss_fn_bce(
            params, state, rng, graphs_padded, net_apply)
        self.assertTrue(new_state == state)

        # calculate expected loss
        a = np.exp(1.)/(np.exp(1.)+np.exp(0.1))
        b = np.exp(1./2)/(np.exp(1./2)+np.exp(1./3))
        loss_expected = -1./6*(3*np.log(a)+np.log(1-a)+np.log(b)+np.log(1-b))
        acc_expected = 2./3
        self.assertAlmostEqual(loss, loss_expected)
        self.assertAlmostEqual(acc, acc_expected)

    def test_init_state(self):
        """Test that init optimizer state is the right class."""
        init_graphs = self.datasets['train'][0]
        with tempfile.TemporaryDirectory() as test_dir:
            _, state, _ = train.init_state(
                self.config, init_graphs, test_dir)
            # Ensure that the initialized state starts from step 0.
            self.assertEqual(state['step'], 0)
            self.assertIsInstance(state['rng'], type(jax.random.PRNGKey(0)))
            self.assertIsInstance(state['opt_state'], tuple)

    def test_numerical_stability(self):
        """Test the minimum losses after 100 steps up to 5 decimal places.

        We want to test the loss looks ok. Also test that result files are
        generated.
        """
        lowest_val_loss = .0
        lowest_val_loss2 = .0
        config = cfg_num.get_config()
        with tempfile.TemporaryDirectory() as test_dir:
            evaluater, lowest_val_loss = train.train_and_evaluate(
                config, test_dir)

            self.assertIsInstance(evaluater.best_state, dict)
            self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
            self.assertIsInstance(lowest_val_loss, (np.float32, np.float64))
            self.assertTrue(os.path.isfile(test_dir + '/REACHED_MAX_STEPS'))

        # reset temp directory
        with tempfile.TemporaryDirectory() as test_dir:

            evaluater, lowest_val_loss2 = train.train_and_evaluate(
                config, test_dir)
            self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
            self.assertTrue(os.path.isfile(test_dir + '/REACHED_MAX_STEPS'))
            self.assertAlmostEqual(lowest_val_loss2, lowest_val_loss, places=6)

    def test_checkpoint_best_loss(self):
        """Test checkpointing of the best loss.

        Test that the best loss is pulled identically in a second run of
        train_and_evaluate. """
        config = cfg_num.get_config()
        with tempfile.TemporaryDirectory() as test_dir:
            evaluater, lowest_val_loss = train.train_and_evaluate(
                config, test_dir)

            self.assertIsInstance(evaluater.best_state, dict)
            self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
            self.assertIsInstance(lowest_val_loss, (np.float32, np.float64))

            evaluater, lowest_val_loss2 = train.train_and_evaluate(
                config, test_dir)
            self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
            self.assertEqual(lowest_val_loss2 - lowest_val_loss, 0.0)

    def test_lowest_loss_upper_bound(self):
        """Test that the lowest loss does not increase with further training.

        First the best loss after 10 steps is calculated, then training is
        continued from checkpoint for 90 more steps until step 100 is reached.
        """
        with tempfile.TemporaryDirectory() as test_dir:
            config = cfg_num.get_config()
            config.num_train_steps_max = 10
            evaluater, lowest_val_loss = train.train_and_evaluate(
                config, test_dir)
            # it was only evaluated at step 10, so check that the step number
            # of the best state is correct
            self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
            self.assertIsInstance(lowest_val_loss, (np.float32, np.float64))

            config.num_train_steps_max = 100
            evaluater, lowest_val_loss2 = train.train_and_evaluate(
                config, test_dir)
            self.assertTrue(lowest_val_loss2 <= lowest_val_loss)

    def test_metrics_logged(self):
        """Test that the right number of evals are computed."""
        config = cfg_num.get_config()
        with tempfile.TemporaryDirectory() as test_dir:
            evaluater, _ = train.train_and_evaluate(
                config, test_dir)
            self.assertCountEqual(
                evaluater.loss_dict.keys(),
                ['train', 'validation', 'test'])
            self.assertEqual(
                len(evaluater.loss_dict['train']),
                config.num_train_steps_max / config.eval_every_steps)
            self.assertIsInstance(
                evaluater.loss_dict['train'][0][0],
                jax.Array)
            self.assertEqual(
                len(evaluater.loss_dict['test'][1][1]),
                2
            )

if __name__ == '__main__':
    unittest.main()

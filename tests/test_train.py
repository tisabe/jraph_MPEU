"""Test the functions in the train module.

Most tests consist of running model training for some steps and then looking at
the resulting metrics and checkpoints.
"""

import tempfile
import unittest

import numpy as np
import jax
from absl import logging

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
            datasets, _, _ = get_datasets(self.config, test_dir)
            self.datasets = datasets

    def test_init_state(self):
        """Test that init optimizer state is the right class."""
        init_graphs = self.datasets['train'][0]
        with tempfile.TemporaryDirectory() as test_dir:
            _, state, _ = train.init_state(
                self.config, init_graphs, test_dir)
            opt_state = state['opt_state']
            # Ensure that the initialized state starts from step 0.
            self.assertEqual(state['step'], 0)
            self.assertIsInstance(state['rng'], type(jax.random.PRNGKey(0)))
            self.assertIsInstance(opt_state, tuple)

    def test_numerical_stability(self):
        """Test the minimum losses after 100 steps up to 5 decimal places.

        We want to test the loss looks ok. Maybe we also want to test that
        things have been written like the metrics.
        """

        config = cfg_num.get_config()
        with tempfile.TemporaryDirectory() as test_dir:
            evaluater, lowest_val_loss = train.train_and_evaluate(
                config, test_dir)

            self.assertIsInstance(evaluater.best_state, dict)
            self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
            self.assertIsInstance(lowest_val_loss, np.float32)

            # reset temp directory
            self.test_dir = tempfile.TemporaryDirectory()

            evaluater, lowest_val_loss2 = train.train_and_evaluate(
                config, test_dir)
            self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
            print(lowest_val_loss2 - lowest_val_loss)
            self.assertAlmostEqual(lowest_val_loss2, lowest_val_loss, places=4)

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
            self.assertIsInstance(lowest_val_loss, np.float32)

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
            self.assertIsInstance(lowest_val_loss, np.float32)

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
            print(evaluater.loss_dict)
            self.assertCountEqual(
                evaluater.loss_dict.keys(),
                ['train', 'validation', 'test'])
            self.assertEqual(
                len(evaluater.loss_dict['train']),
                config.num_train_steps_max / config.eval_every_steps)
            self.assertIsInstance(
                evaluater.loss_dict['train'][0][0],
                jax.numpy.DeviceArray)
            self.assertEqual(
                len(evaluater.loss_dict['test'][1][1]),
                2
            )

if __name__ == '__main__':
    unittest.main()

import ml_collections
import numpy as np
import tempfile
import unittest
import jax
import haiku as hk
import optax
from configs import test_numerics as cfg_num

import train
from input_pipeline import get_datasets

class TestTrain(unittest.TestCase):
    """Test train.py methods and classes."""
    def setUp(self):
        from configs import default_test as cfg
        self.config = cfg.get_config()
        self.config.limit_data = 100
        self.assertEqual(self.config.batch_size, 32)
        # get testing datasets
        datasets, _, _, _ = get_datasets(self.config)
        self.datasets = datasets
        self.datasets_train_val = {
            'train': self.datasets['train'].data[:],
            'validation': self.datasets['validation'].data[:]}
        self.test_dir = tempfile.TemporaryDirectory()

    def test_init_state(self):
        """Test that init optimizer state is the right class."""
        init_graphs = next(self.datasets['train'])
        _, state, _ = train.init_state(
                self.config, init_graphs, self.test_dir.name)
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
        evaluater, lowest_val_loss = train.train_and_evaluate(
            config, self.test_dir.name)
        
        self.assertIsInstance(evaluater.best_state, dict)
        self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
        self.assertIsInstance(lowest_val_loss, np.float32)
        
        # reset temp directory
        self.test_dir = tempfile.TemporaryDirectory()
        
        evaluater, lowest_val_loss2 = train.train_and_evaluate(
            config, self.test_dir.name)
        self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
        print(lowest_val_loss2 - lowest_val_loss)
        self.assertAlmostEqual(lowest_val_loss2, lowest_val_loss, places=4)

    def test_checkpoint_best_loss(self):
        """Test that the best loss is pulled identically in a second run of train_and_evaluate. """
        config = cfg_num.get_config()
        evaluater, lowest_val_loss = train.train_and_evaluate(
            config, self.test_dir.name)
        
        self.assertIsInstance(evaluater.best_state, dict)
        self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
        self.assertIsInstance(lowest_val_loss, np.float32)
        
        evaluater, lowest_val_loss2 = train.train_and_evaluate(
            config, self.test_dir.name)
        self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
        self.assertEqual(lowest_val_loss2 - lowest_val_loss, 0.0)

    def test_lowest_loss_upper_bound(self):
        self.test_dir = tempfile.TemporaryDirectory()
        config = cfg_num.get_config()
        evaluater, lowest_val_loss = train.train_and_evaluate(
            config, self.test_dir.name)
        
        self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
        self.assertIsInstance(lowest_val_loss, np.float32)
        
        config.num_train_steps_max = 100
        evaluater, lowest_val_loss2 = train.train_and_evaluate(
            config, self.test_dir.name)
        self.assertEqual(evaluater.best_state['step'], config.num_train_steps_max)
        print(lowest_val_loss2 - lowest_val_loss)
        self.assertTrue(lowest_val_loss2 <= lowest_val_loss)

    def test_metrics_logged(self):
        self.test_dir = tempfile.TemporaryDirectory()
        config = cfg_num.get_config()
        evaluater, lowest_val_loss = train.train_and_evaluate(
            config, self.test_dir.name)
        print(evaluater.loss_dict)
        self.assertCountEqual(evaluater.loss_dict.keys(), ['train', 'validation', 'test'])
        self.assertEqual(len(evaluater.loss_dict['train']), 
                config.num_train_steps_max / config.eval_every_steps)
    
if __name__ == '__main__':
    unittest.main()

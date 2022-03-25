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
        self.test_dir = tempfile.TemporaryFile()

    def test_init_state(self):
        """Test that init optimizer state is the right class."""
        init_graphs = next(self.datasets['train'])
        workdir = 'results/test/checkpoint'
        _, state, _ = train.init_state(
                self.config, init_graphs, workdir)
        print(type(state['params']))
        opt_state = state['opt_state']
        # Ensure that the initialized state starts from step 0.
        self.assertEqual(state['step'], 0)
        self.assertIsInstance(state['rng'], type(jax.random.PRNGKey(0)))
        print(type(opt_state))
        #self.assertIsInstance(opt_state, optax.GradientTransformation)
        # TODO: find out the right type to check
        #self.assertIsInstance(state.params, hk._src.data_structures.FlatMap)

    def test_numerical_stability(self):
        """Test the minimum losses after 100 steps up to 5 decimal places.
        
        We want to test the loss looks ok. Maybe we also want to test that
        things have been written like the metrics.
        """
        
        config = cfg_num.get_config()
        best_state = train.train_and_evaluate(
            config, self.test_dir)
        # self.assertTrue(isinstance(best_state, train.train.Evaluater))
        print(best_state)
        print(type(best_state))
        self.assertFalse(True)
        # best_state, min_loss_new = train.train(config, self.datasets_train_val)
        # self.assertAlmostEqual(min_loss, min_loss_new, places=5)
    
if __name__ == '__main__':
    unittest.main()

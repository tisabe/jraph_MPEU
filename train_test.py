import ml_collections
import numpy as np
import unittest
import jax
import haiku as hk
import optax

import train
from input_pipeline import get_datasets

class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        from configs import default_test as cfg
        self.config = cfg.get_config()
        self.config.limit_data = 100
        self.assertEqual(self.config.batch_size, 32)
        # get testing datasets
        rng = jax.random.PRNGKey(42)
        rng, data_rng = jax.random.split(rng)
        datasets, _, _, _ = get_datasets(self.config, data_rng)
        self.datasets = datasets

    def test_init_state(self):
        init_graphs = next(self.datasets['train'])
        state = train.init_state(self.config, init_graphs)
        print(type(state.params))
        self.assertEqual(state.step, 0)
        self.assertFalse(state.apply_fn is None)
        self.assertIsInstance(state.tx, optax.GradientTransformation)
        # TODO: find out the right type to check
        #self.assertIsInstance(state.params, hk._src.data_structures.FlatMap)

    def test_numerical_stability(self):
        from configs import test_numerics as cfg_num
        
    
if __name__ == '__main__':
    unittest.main()

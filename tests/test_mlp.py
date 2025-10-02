"""Test MLP."""
import unittest

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from jraph_MPEU.models.mlp import MLP


class UnitTests(unittest.TestCase):
    """Unit and integration test functions in models/mlp.py."""
    def test_batch_norm(self):
        """Test the _build_mlp function."""
        hidden_sizes = [16, 16, 8]
        # test with relu activation, and identity weight-matrices,
        # with gain of 2, i.e. multiplying with 2 every layer
        hk_init = hk.initializers.Identity(2.0)
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        def mlp(x):
            return MLP(
                name='test_name',
                output_sizes=hidden_sizes,
                use_layer_norm=False,
                use_batch_norm=True,
                dropout_rate=.0,
                activation=jax.nn.relu,
                with_bias=False,
                activate_final=False,
                w_init=hk_init,
                batch_norm_decay=0.999,
                is_training=True
            )(x)

        mlp = hk.transform_with_state(mlp)
        init_inputs = jnp.zeros((1, 10))
        params, state = mlp.init(init_rng, init_inputs)

        inputs = np.array([
            [-4, -2, 0, 1, 2, 3, 4, 5, 6, 7],
            [-2, 0, 0, -1, 0, 1, 2, 3, 4, 5]
        ])
        outputs, state = mlp.apply(params, state, rng, inputs)
        expected_outputs = np.array([
            [0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., -1., -1., -1., -1., -1.],
        ])
        np.testing.assert_allclose(outputs, expected_outputs, atol=6)

        # create new mlp with is_training=True to test batch_norm
        def mlp_new(x):
            return MLP(
                name='test_name',
                output_sizes=hidden_sizes,
                use_layer_norm=False,
                use_batch_norm=True,
                dropout_rate=.0,
                activation=jax.nn.relu,
                with_bias=False,
                activate_final=False,
                w_init=hk_init,
                batch_norm_decay=0.999,
                is_training=False
            )(x)

        mlp = hk.transform_with_state(mlp_new)
        outputs, state = mlp.apply(params, state, rng, inputs)
        np.testing.assert_allclose(outputs, expected_outputs, atol=6)

if __name__ == '__main__':
    unittest.main()

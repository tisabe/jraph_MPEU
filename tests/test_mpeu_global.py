"""Test MPEU_uq (MPEU with uncertainty quantification) model.
"""
import unittest

from parameterized import parameterized
import numpy as np
import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import ml_collections

from jraph_MPEU.models.mpeu_global import MPEU_global
from jraph_MPEU_configs.aflow_class_test import get_config as get_class_config


def _get_random_graph(rng, z_max) -> jraph.GraphsTuple:
    """Return a random graphsTuple with n_node from 1 to 10 and intergers up to
    z_max as node features."""
    n_node = rng.integers(1, 10, (1,))
    n_edge = rng.integers(1, 20, (1,))
    graph = jraph.GraphsTuple(
        nodes=rng.integers(0, z_max, (n_node[0],)),
        edges=rng.random((n_edge[0], 3)),
        senders=rng.integers(0, n_node[0], (n_edge[0],)),
        receivers=rng.integers(0, n_node[0], (n_edge[0],)),
        n_node=n_node,
        n_edge=n_edge,
        globals=np.array([[0, 1]]))
    return graph


def _get_random_graph_batch(rng, n_graphs, z_max) -> jraph.GraphsTuple:
    """Return a batch of random graphs."""
    graphs = []
    for _ in range(n_graphs):
        graph = _get_random_graph(rng, z_max)
        graphs.append(graph)
    return jraph.batch(graphs)


class TestModelFunctions(unittest.TestCase):
    """Unit and integration test functions in models.py."""
    def setUp(self):
        n_node = jnp.arange(3, 11) # make graphs with 3 to 10 nodes
        n_edge = jnp.arange(4, 12) # make graphs with 4 to 11 edges
        total_n_node = jnp.sum(n_node)
        total_n_edge = jnp.sum(n_edge)
        n_graph = n_node.shape[0]
        feature_dim = 10
        self.np_rng = np.random.default_rng(seed=7)

        self.graphs = jraph.GraphsTuple(
            n_node=n_node,
            n_edge=n_edge,
            senders=jnp.zeros(total_n_edge, dtype=jnp.int32),
            receivers=jnp.ones(total_n_edge, dtype=jnp.int32),
            nodes=jnp.ones((total_n_node, feature_dim)),
            edges=jnp.zeros((total_n_edge, feature_dim)),
            globals=jnp.zeros((n_graph, feature_dim)),
        )
        self.config = ml_collections.ConfigDict()
        self.config.batch_size = 32
        self.config.latent_size = 64
        self.config.use_layer_norm = False
        self.config.use_batch_norm = False
        self.config.message_passing_steps = 3
        self.config.mlp_depth = 2
        self.hk_init = hk.initializers.Identity()
        self.config.aggregation_message_type = 'sum'
        self.config.aggregation_readout_type = 'sum'
        self.config.k_max = 150
        self.config.delta = 0.1
        self.config.mu_min = 0.0
        self.config.max_atomic_number = 5
        self.config.dropout_rate = 0.0
        self.config.label_type = 'scalar'
        self.config.activation_name = 'shifted_softplus'
        self.activation_names = ['shifted_softplus', 'relu', 'swish']
        self.config.global_readout_mlp_layers = 2

    @parameterized.expand([
        'activation_name',
        'aggregation_message_type',
        'aggregation_readout_type'])
    def test_raises_activation_error(self, config_field):
        """Test that an error is raised, when wrong string is used in
        config_field."""
        with self.assertRaises(KeyError):
            self.config[config_field] = 'test'
            MPEU_global(self.config, True)

    def test_random_graph_batch(self):
        n_graphs = 100

        graph = _get_random_graph_batch(
            self.np_rng, n_graphs, self.config.max_atomic_number)
        self.config.hk_init = None
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        net_fn = MPEU_global(self.config, True)
        net = hk.transform(net_fn)
        params = net.init(init_rng, graph)

        graph_pred = net.apply(params, rng, graph)
        prediction = graph_pred.globals
        self.assertTupleEqual(prediction.shape, (n_graphs, 1))


if __name__ == '__main__':
    unittest.main()

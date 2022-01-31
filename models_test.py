import numpy as np
import jax
import jax.numpy as jnp
import jraph
import haiku as hk

from absl import flags
import ml_collections
from ml_collections import config_flags

import unittest

from models import GNN, shifted_softplus


class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        n_node = jnp.arange(3, 11) # make graphs with 3 to 10 nodes
        n_edge = jnp.arange(4, 12) # make graphs with 4 to 11 edges
        total_n_node = jnp.sum(n_node)
        total_n_edge = jnp.sum(n_edge)
        n_graph = n_node.shape[0]
        feature_dim = 10

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
        self.config.message_passing_steps = 3
        self.hk_init = hk.initializers.Identity()
        self.config.aggregation_message_type = 'sum'
        self.config.aggregation_readout_type = 'sum'
        self.config.k_max = 150
        self.config.delta = 0.1
        self.config.mu_min = 0.0
        self.config.max_atomic_number = 5
        
    def test_GNN_output_zero_graph(self):
        n_node = 10
        n_edge = 4
        n_node_features = 1
        n_edge_features = 1
        init_graphs = jraph.GraphsTuple(
            nodes=jnp.ones((n_node))*5,
            edges=jnp.ones((n_edge)),
            senders=jnp.array([0,0,1,2]),
            receivers=jnp.array([1,2,0,0]),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_edge]),
            globals=None
        )
        label = np.array([1.0])

        # graph with zeros as node features and a self edge for every node
        graph_zero = jraph.GraphsTuple(
            nodes=jnp.zeros((n_node)),
            edges=jnp.ones((n_node)),
            senders=jnp.arange(0,n_node),
            receivers=jnp.arange(0,n_node),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_node]),
            globals=None
        )
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        
        self.config.hk_init = hk.initializers.Identity()
        net_fn = GNN(self.config)
        net = hk.without_apply_rng(hk.transform(net_fn))
        params = net.init(init_rng, init_graphs) # create weights etc. for the model
        
        graph_pred = net.apply(params, graph_zero)
        prediction = graph_pred.globals
        
        sp = shifted_softplus # shorten shifted softplus function
        h0p = sp(sp(sp(sp(1)))) + 1
        h0pp = h0p + sp(sp(sp(sp(h0p)))*h0p)
        h0ppp = h0pp + sp(sp(sp(sp(h0pp)))*h0pp)
        label = n_node*sp(h0ppp)
        
        np.testing.assert_array_equal(prediction.shape, (1,1))
        self.assertAlmostEqual(label, prediction[0,0], places=5)

        return 0




if __name__ == '__main__':
    unittest.main()
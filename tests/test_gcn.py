"""Test our GCN Model.
"""
import unittest

import numpy as np
import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import ml_collections

from jraph_MPEU.models.mlp import(
    shifted_softplus,
    MLP
)
from jraph_MPEU.models.gcn import GCN


def _get_random_graph(max_n_graph=10,
                      include_node_features=True,
                      include_edge_features=True,
                      include_globals=True):
    """Get a batch of graphs.
    
    node features are shape (sum(n_node), 4), if included
    edge features are shape (sum(n_edge), 3), if included
    global features are shape (n_graph, 5), if included
    """
    n_graph = np.random.randint(1, max_n_graph + 1) # max_n_graph graphs
    n_node = np.random.randint(0, 10, n_graph) # 1-D array of # of numbers
    n_edge = np.random.randint(0, 20, n_graph) # 1-D array of # of edges
    # We cannot have any edges if there are no nodes.
    n_edge[n_node == 0] = 0

    senders = []
    receivers = []
    offset = 0
    for n_node_in_graph, n_edge_in_graph in zip(n_node, n_edge):
        # loop over different (sub-) graphs
        if n_edge_in_graph != 0:
            # append n_edge_in_graph random senders to senders
            senders += list(
                np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset)
            receivers += list(
                np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset)
        offset += n_node_in_graph
    if include_globals:
        global_features = jnp.asarray(np.random.random(size=(n_graph, 5)))
    else:
        global_features = None
    if include_node_features:
        nodes = jnp.asarray(np.random.random(size=(np.sum(n_node), 4)))
    else:
        nodes = None

    if include_edge_features:
        edges = jnp.asarray(np.random.random(size=(np.sum(n_edge), 3)))
    else:
        edges = None
    return jraph.GraphsTuple(
        n_node=jnp.asarray(n_node),
        n_edge=jnp.asarray(n_edge),
        nodes=nodes,
        edges=edges,
        globals=global_features,
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers))


class TestModelFunctions(unittest.TestCase):
    """Unit and integration test functions in models.py."""
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
        self.config.use_layer_norm = False
        self.config.message_passing_steps = 3
        self.config.global_readout_mlp_layers = 0
        self.config.mlp_depth = 2
        self.hk_init = hk.initializers.Identity()
        self.config.aggregation_message_type = 'sum'
        self.config.aggregation_readout_type = 'sum'
        self.config.max_atomic_number = 5
        self.config.dropout_rate = 0.0
        self.config.label_type = 'scalar'
        self.config.activation_name = 'shifted_softplus'
        self.activation_names = ['shifted_softplus', 'relu', 'swish']

    def test_GNN_output_zero_graph(self):
        """We want to test the basic equation of GCN:
        
        H^{l+1}=\sigma (\tilde{d}^{1/2}\tilde{A}\tilde{d}^{1/2} H^l W^l)
        """
        graph = _get_random_graph(max_n_graph=2)
        print(graph)

        n_node = 10
        n_edge = 4
        init_graphs = jraph.GraphsTuple(
            nodes=jnp.ones((n_node))*5,
            edges=jnp.ones((n_edge)),
            senders=jnp.array([0, 0, 1, 2]),
            receivers=jnp.array([1, 2, 0, 0]),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_edge]),
            globals=None
        )
        label = np.array([1.0])

        # graph with zeros as node features and a self edge for every node
        graph_zero = jraph.GraphsTuple(
            nodes=jnp.zeros((n_node)),
            edges=jnp.ones((n_node)),
            senders=jnp.arange(0, n_node),
            receivers=jnp.arange(0, n_node),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_node]),
            globals=None
        )
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        self.config.hk_init = hk.initializers.Identity()
        for activation_name in self.activation_names:
            if activation_name == 'shifted_softplus':
                sp = shifted_softplus
            elif activation_name == 'swish':
                sp = jax.nn.swish
            elif activation_name == 'relu':
                sp = jax.nn.relu
            self.config.activation_name = activation_name
            net_fn = GCN(self.config)
            net = hk.transform(net_fn)
            params = net.init(init_rng, init_graphs) # create weights etc. for the model

            graph_pred = net.apply(params, rng, graph_zero)
            prediction = graph_pred.globals

            # the following is the analytical expected result
            h0p = sp(sp(sp(sp(1.0)))) + 1.0
            h0pp = h0p + sp(sp(sp(sp(h0p)))*h0p)
            h0ppp = h0pp + sp(sp(sp(sp(h0pp)))*h0pp)
            label = n_node*sp(h0ppp)

            np.testing.assert_array_equal(prediction.shape, (1, 1))
            self.assertAlmostEqual(label, prediction[0, 0], places=5)


if __name__ == '__main__':
    unittest.main()

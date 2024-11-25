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

def _get_random_adjacency(n_node, symmetric=False, self_edges=False):
    """Return a random adjacency matrix, with n_node nodes."""
    adjacency = np.random.randint(0, 2, (n_node, n_node))
    if symmetric:
        # symmetrize adjacency matrix, this works because entries are 0 or 1
        adjacency = np.maximum(adjacency, adjacency.transpose())
    if self_edges:
        np.fill_diagonal(adjacency, 1) # happens in place
    else:
        np.fill_diagonal(adjacency, 0) # happens in place
    return adjacency


def _get_random_graph(max_n_graph=10,
                      include_node_features=True,
                      include_edge_features=True,
                      include_globals=True,
                      multi_edges=True):
    """Get a batch of graphs with random numbers of nodes and edges.
    
    node features are shape (sum(n_node), 4), if included
    edge features are shape (sum(n_edge), 3), if included
    global features are shape (n_graph, 5), if included

    If multi_edges=True, there can be multiple edges between one pair of nodes.
    """
    n_graph = np.random.randint(1, max_n_graph + 1) # max_n_graph graphs
    n_node = np.random.randint(0, 10, n_graph) # 1-D array of # of nodes
    if multi_edges:
        n_edge = np.random.randint(0, 20, n_graph) # 1-D array of # of edges

        # We cannot have any edges if there are no nodes.
        n_edge[n_node == 0] = 0

        senders = []
        receivers = []
        offset = 0
        for n_node_in_graph, n_edge_in_graph in zip(n_node, n_edge):
            # loop over different (sub-) graphs
            if n_edge_in_graph != 0:
                if multi_edges:
                    # append n_edge_in_graph random senders to senders
                    senders += list(np.random.randint(
                        0, n_node_in_graph, n_edge_in_graph) + offset)
                    receivers += list(np.random.randint(
                        0, n_node_in_graph, n_edge_in_graph) + offset)
            offset += n_node_in_graph
    else:
        # generate adjacency matrix to get n_edge, senders and receivers
        n_edge = []
        senders = []
        receivers = []
        offset = 0
        for n_node_in_graph in n_node:
            n_edge.append(0)
            adjacency = _get_random_adjacency(
                n_node_in_graph, symmetric=True, self_edges=False)
            for row_i, row in enumerate(adjacency):
                for col_i, val in enumerate(row):
                    if val==1:
                        receivers.append(row_i + offset)
                        senders.append(col_i + offset)
                        n_edge[-1] += 1 # update the edge counter
            offset += n_node_in_graph
        n_edge = np.array(n_edge)

    if include_globals:
        global_features = jnp.asarray(np.random.random(size=(n_graph, 5)))
    else:
        global_features = None
    if include_node_features:
        nodes = jnp.asarray(np.random.randint(1, 5, size=np.sum(n_node)))
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


def _get_adjacency_from_lists(n_node, senders, receivers):
    """Return the graph adjacency matrix from list of senders and receivers."""
    adjacency = np.zeros((n_node, n_node), int)
    for sender, receiver in zip(senders, receivers):
        adjacency[receiver, sender] = 1
    return adjacency


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
        self.config.message_passing_steps = 1
        self.config.global_readout_mlp_layers = 0
        self.hk_init = hk.initializers.Identity()
        self.config.aggregation_message_type = 'sum'
        self.config.aggregation_readout_type = 'sum'
        self.config.max_atomic_number = 5
        self.config.dropout_rate = 0.0
        self.config.label_type = 'scalar'
        self.config.activation_name = 'shifted_softplus'

    def test_GNN_output_zero_graph(self):
        """We want to test the basic equation of GCN:
        
        H^{l+1}=\sigma (\tilde{d}^{1/2}\tilde{A}\tilde{d}^{1/2} H^l W^l)
        """
        np.random.seed(4)
        graph = _get_random_graph(max_n_graph=2, multi_edges=False)
        adjacency = _get_adjacency_from_lists(
            sum(graph.n_node), graph.senders, graph.receivers)
        # ensure there are self edges
        adjacency = np.maximum(adjacency, np.eye(adjacency.shape[0]))
        # calculate the D_ii matrix that denotes the incoming edge degree
        # this is already inverse squareroot
        degree_matrix = np.diag(np.sum(adjacency, axis=-1)**(-.5))

        x = np.reshape(graph.nodes, (-1)) # original node features
        h_0 = jax.nn.one_hot(x, self.config.max_atomic_number)
        latent_size = 8
        self.config.latent_size = latent_size
        #weights_0 = np.random.random_sample(
        #    size=(np.shape(h_0)[-1], latent_size))
        weights_0 = np.eye(np.shape(h_0)[-1], latent_size)
        h_1 = degree_matrix @ adjacency @ degree_matrix @ h_0 @ weights_0
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        self.config.hk_init = hk.initializers.Identity()

        net_fn = GCN(self.config)
        net = hk.transform(net_fn)
        params = net.init(init_rng, graph) # create weights etc. for the model

        graph_pred = net.apply(params, rng, graph)
        np.testing.assert_allclose(graph_pred.nodes, h_1, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()

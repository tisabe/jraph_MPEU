"""Test our GNN Models.


Right now we have a test with 0/1s on a graph but we only use self edges
since otherwise the complexity means it's hard to figure out what the right
answer is on paper to check we're getting the right answer.
"""
import unittest

import numpy as np
import jax
import jax.numpy as jnp
import jraph
import haiku as hk

import ml_collections

from models import (
    GNN,
    shifted_softplus,
    get_edge_update_fn,
    get_edge_embedding_fn,
    get_node_embedding_fn,
    get_node_update_fn
)


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
        self.config.message_passing_steps = 3
        self.hk_init = hk.initializers.Identity()
        self.config.aggregation_message_type = 'sum'
        self.config.aggregation_readout_type = 'sum'
        self.config.k_max = 150
        self.config.delta = 0.1
        self.config.mu_min = 0.0
        self.config.max_atomic_number = 5

    def test_GNN_output_zero_graph(self):
        """Test the forward pass of the MPNN on a graph with zeroes as features.

        Test for a single graph with 10 nodes and 4 edges, zeroes as node
        feature vectors and ones as edge feature vectors.
        The forward passed graph has only self edges and no globals.

        The GNN function uses:
        shifted_softplus
        edge_embedding_fn
        node_embedding_fn
        edge_update_fn
        node_update_fn
        readout_global_fn
        readout_node_update_fn

        TODO (Tim): test non-self edges
        """
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

        self.config.latent_size = 64
        self.config.message_passing_steps = 3
        self.config.hk_init = hk.initializers.Identity()
        net_fn = GNN(self.config)
        net = hk.without_apply_rng(hk.transform(net_fn))
        params = net.init(init_rng, init_graphs) # create weights etc. for the model

        graph_pred = net.apply(params, graph_zero)
        prediction = graph_pred.globals

        # the following is the analytical expected result
        sp = shifted_softplus  # shorten shifted softplus function
        h0p = sp(sp(sp(sp(1)))) + 1
        h0pp = h0p + sp(sp(sp(sp(h0p)))*h0p)
        h0ppp = h0pp + sp(sp(sp(sp(h0pp)))*h0pp)
        label = n_node*sp(h0ppp)

        np.testing.assert_array_equal(prediction.shape, (1, 1))
        self.assertAlmostEqual(label, prediction[0, 0], places=5)


    def test_edge_update_fn(self):
        """Test the expected edge and message feature vector in the edge update.

        Pass a single edge into the edge update function and compare with
        expected output.
        """
        latent_size = 10
        hk_init = hk.initializers.Identity()
        edge_update_fn = get_edge_update_fn(latent_size, hk_init)

        # Sent node features corresponding to the edge.
        sent_attributes = jnp.arange(0, 4)
        # Received node features corresponding to the edge.
        received_attributes = jnp.arange(4, 8)
        # Edge feature vector, single edge with dimension 2.
        edge = jnp.arange(8, 10)
        # Message feature vector, single message with dimension 15.
        message = jnp.arange(0, 15)
        # Create structured edge message dictionary.
        edge_message = {'edges': edge, 'messages': message}
        # Arbitrary globals, for call signature.
        graph_globals = jnp.ones((11))

        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        # Create pure callable with haiku to initialize later.
        hk_edge_update_fn = hk.transform(edge_update_fn)
        # We do not use dropout so we don't need rngs.
        hk_edge_update_fn = hk.without_apply_rng(hk_edge_update_fn)
        # Initialize weights and paramters with haiku.
        # We use a different to initialize, this tests the ability of the
        # function to use different inputs.
        params = hk_edge_update_fn.init(
            init_rng,
            {'edges': jnp.zeros((2)), 'message': jnp.zeros(15)},
            jnp.zeros((4)), jnp.zeros((4)),
            jnp.zeros((11)))

        edge_message_update = hk_edge_update_fn.apply(
            params, edge_message, sent_attributes, received_attributes,
            graph_globals)

        edge_updated = edge_message_update['edges']
        message_updated = edge_message_update['messages']

        edge_expected = shifted_softplus(jnp.arange(0, 10))
        np.testing.assert_allclose(edge_updated, edge_expected)

        # test the result of message calculation
        node_message_expected = jnp.pad(sent_attributes, (0, 6))
        edge_message_expected = shifted_softplus(shifted_softplus(edge_expected))
        message_expected = jnp.multiply(node_message_expected, edge_message_expected)

        np.testing.assert_allclose(message_updated, message_expected)
        return 0

    def test_edge_embedding_fn(self):
        """Test the edge embedding function by comparing the expected result.
        """
        latent_size = 64
        edges = jnp.ones((1))
        k_max = 150
        k_vec = jnp.arange(0, k_max)
        delta = 0.1
        mu_min = 0.2
        fun = get_edge_embedding_fn(latent_size, k_max, delta, mu_min)
        embedding = fun(edges)

        embedding_expected = [
            jnp.exp(-jnp.square(edges[0]-(-mu_min+k_vec*delta))/delta)]

        np.testing.assert_allclose(embedding['edges'], embedding_expected, atol=6)
        np.testing.assert_array_equal(
            embedding['messages'],
            jnp.zeros((1, latent_size)))

    def test_node_embedding_fn(self):
        '''Test the node embedding function'''
        latent_size = 16
        max_atomic_number = 5
        hk_init = hk.initializers.Identity()
        node_embedding_fn = get_node_embedding_fn(latent_size, max_atomic_number, hk_init)
        node_embedding_fn = hk.testing.transform_and_run(node_embedding_fn)

        nodes = jnp.arange(1, 9) # simulating atomic numbers from 1 to 8
        #print(f'Input nodes: {nodes}')

        nodes_embedded = node_embedding_fn(nodes)
        #print(f'Embedded nodes: {nodes_embedded}')

        np.testing.assert_array_equal(
            [len(nodes), latent_size],
            nodes_embedded.shape) # check shape of output

        sum_nodes_expected = min(max_atomic_number-1, len(nodes))
        self.assertEqual(int(jnp.sum(nodes_embedded)), sum_nodes_expected)

    def test_node_update_fn(self):
        '''Test the node update function.

        This is equation 9 in PB Jorgensen paper.
        '''
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        latent_size = 16
        hk_init = hk.initializers.Identity()
        node_update_fn = get_node_update_fn(latent_size, hk_init)

        num_nodes = 10
        nodes = jnp.ones((num_nodes, latent_size))
        sent_message = {
            'edges': None,
            'messages': jnp.ones((num_nodes, latent_size))*2
        }
        received_message = {
            'edges': None,
            'messages': jnp.ones((num_nodes, latent_size))*3
        }
        global_attributes = None

        node_update_fn = hk.transform(node_update_fn)
        node_update_fn = hk.without_apply_rng(node_update_fn)
        params = node_update_fn.init(
            init_rng,
            nodes, sent_message, received_message, global_attributes
        )

        nodes_updated = node_update_fn.apply(
            params,
            nodes, sent_message, received_message, global_attributes
        )
        np.testing.assert_allclose(
            nodes_updated,
            shifted_softplus(received_message['messages']) + nodes)

if __name__ == '__main__':
    unittest.main()

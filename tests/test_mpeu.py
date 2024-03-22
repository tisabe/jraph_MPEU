"""Test our GNN Models.


Right now we have a test with 0/1s on a graph but we only use self edges
since otherwise the complexity means it's hard to figure out what the right
answer is on paper to check we're getting the right answer.
"""
import unittest

from parameterized import parameterized
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
from jraph_MPEU.models.mpeu import (
    MPEU,
    _get_edge_update_fn,
    _get_edge_embedding_fn,
    _get_node_update_fn,
    _get_node_embedding_fn,
    _get_readout_node_update_fn,
    #get_readout_global_fn
)
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
        globals=None)
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
        self.config.global_readout_mlp_layers = 0
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

    @parameterized.expand([
        'activation_name',
        'aggregation_message_type',
        'aggregation_readout_type'])
    def test_raises_activation_error(self, config_field):
        """Test that an error is raised, when wrong string is used in
        config_field."""
        with self.assertRaises(ValueError):
            self.config[config_field] = 'test'
            MPEU(self.config, True)

    def test_build_extra_layer(self):
        """Test building the MPEU with extra layer and scalar label type."""
        n_graphs = 100
        graph = _get_random_graph_batch(
            self.np_rng, n_graphs, self.config.max_atomic_number)

        self.config.hk_init = None
        self.config.global_readout_mlp_layers = 2
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        net_fn = MPEU(self.config, True)
        net = hk.transform(net_fn)
        params = net.init(init_rng, graph)

        graph_pred = net.apply(params, rng, graph)
        prediction = graph_pred.globals
        self.assertTupleEqual(prediction.shape, (n_graphs, 1))

    def test_mpeu_non_negative_output(self):
        """Test that when the label type is 'scalar_non_negative', the output
        of the MPEU is the same as before, but with ReLU on output."""
        n_graphs = 100
        graph = _get_random_graph_batch(
            self.np_rng, n_graphs, self.config.max_atomic_number)

        self.config.hk_init = None
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        net_fn = MPEU(self.config, True)
        net = hk.transform(net_fn)
        params = net.init(init_rng, graph)

        graph_pred = net.apply(params, rng, graph)
        prediction = graph_pred.globals

        # now create and predict with model with different label_type
        self.config.label_type = 'scalar_non_negative'
        net_fn = MPEU(self.config, True)
        net = hk.transform(net_fn)
        params = net.init(init_rng, graph)

        graph_pred = net.apply(params, rng, graph)
        prediction_relu = graph_pred.globals

        np.testing.assert_array_equal(prediction_relu, jax.nn.relu(prediction))


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
        """

        n_node = 10
        n_edge = 4
        init_graphs = jraph.GraphsTuple(
            nodes=jnp.ones((n_node))*5,
            edges=jnp.ones((n_edge, 3)),
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
            edges=jnp.ones((n_node, 3)),
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
            net_fn = MPEU(self.config, True)
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

    def test_GNN_standard_graph_output(self):
        """Test the output of the GNN with a graph with representative edges.

        The graph consists of 3 nodes and 4 edges:
        o <-> o <- o
         --------->  (this edge goes from node 0 to node 2).
        Each node has its index as a feature, e.g. node 1's feature is [1].
        The edge features go from 1 to 4.
        The weight matrices are initialized to all constant values of
        1 / latent_size, to make the analytical result easier to calculate.
        """
        self.config.latent_size = 16
        latent_size = self.config.latent_size
        self.config.delta = 10
        self.config.k_max = 16
        self.config.message_passing_steps = 1
        self.config.max_atomic_number = 5
        self.config.hk_init = hk.initializers.Constant(1 / latent_size)
        self.config.aggregation_message_type = 'sum'
        self.config.aggregation_readout_type = 'sum'

        n_node = 10
        n_edge = 4
        init_graphs = jraph.GraphsTuple(
            nodes=jnp.ones((n_node))*5,
            edges=jnp.ones((n_edge, 3)),
            senders=jnp.array([0, 0, 1, 2]),
            receivers=jnp.array([1, 2, 0, 0]),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_edge]),
            globals=None
        )

        n_node = jnp.array([3])
        n_edge = jnp.array([4])
        node_features = jnp.array([0, 1, 2])
        edge_features = jnp.array([[0, 1], [2, 0], [3, 0], [0, 4]])
        edge_dists = jnp.sqrt(jnp.sum(edge_features**2, axis=1))
        senders = jnp.array([0, 1, 2, 0])
        receivers = jnp.array([1, 0, 1, 2])

        graph = jraph.GraphsTuple(
            nodes=node_features,
            edges=edge_features,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals=None
        )

        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        for activation_name in self.activation_names:
            if activation_name == 'shifted_softplus':
                activation_fn = shifted_softplus
            elif activation_name == 'swish':
                activation_fn = jax.nn.swish
            elif activation_name == 'relu':
                activation_fn = jax.nn.relu
            self.config.activation_name = activation_name
            net_fn = MPEU(self.config, True)
            net = hk.transform(net_fn)
            # Create weights for the model
            params = net.init(init_rng, init_graphs)

            graph_pred = net.apply(params, rng, graph)
            prediction = graph_pred.globals

            # Here, we calculate the expected result, starting with the embeddings.
            nodes_embedded = jnp.ones((n_node[0], latent_size))/latent_size
            edge_embedding_fn = _get_edge_embedding_fn(
                latent_size,
                self.config.k_max,
                self.config.delta,
                self.config.mu_min)
            edges_embedded = edge_embedding_fn(edge_dists)['edges']
            edges_embedded = np.array(edges_embedded)

            # The updated edges.
            edge_factor = (np.sum(edges_embedded, axis=-1) + 2)/latent_size
            edge_factor = activation_fn(edge_factor)*2
            edge_factor = np.array(edge_factor)

            edge_updated = jnp.ones((n_edge[0], latent_size))
            for i in range(n_edge[0]):
                edge_updated = edge_updated.at[i].set(
                    edge_updated[i] * float(edge_factor[i]))

            # The edge-wise message.
            messages = activation_fn(edge_updated)
            messages = activation_fn(messages)/latent_size

            # Node-wise message
            message_node = jnp.zeros((n_node[0], latent_size))
            message_node = message_node.at[0].set(messages[1])
            message_node = message_node.at[1].set(messages[0] + messages[2])
            message_node = message_node.at[2].set(messages[3])

            # State-transition
            nodes_updated = nodes_embedded + activation_fn(message_node)

            # Readout
            nodes_readout = activation_fn(nodes_updated[:, 0])/2
            prediction_expected = jnp.sum(nodes_readout)

            np.testing.assert_allclose(
                edge_updated, graph_pred.edges['edges'], rtol=1e-6)
            self.assertAlmostEqual(prediction_expected, prediction, places=5)

    def test_output_size_class(self):
        """Test that the output dimensions of the GNN function are as intended.
        The output shape should be:
        [batch_size, 2] for binary classification task.
        """
        n_node = 10
        n_edge = 4
        init_graphs = jraph.GraphsTuple(
            nodes=jnp.ones((n_node))*5,
            edges=jnp.ones((n_edge, 4)),
            senders=jnp.array([0, 0, 1, 2]),
            receivers=jnp.array([1, 2, 0, 0]),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_edge]),
            globals=None
        )

        n_node = jnp.array([3])
        n_edge = jnp.array([4])
        node_features = jnp.array([0, 1, 2])
        edge_features = jnp.array([[0, 1], [2, 0], [3, 0], [0, 4]])
        senders = jnp.array([0, 1, 2, 0])
        receivers = jnp.array([1, 0, 1, 2])

        graph = jraph.GraphsTuple(
            nodes=node_features,
            edges=edge_features,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals=None
        )
        graphs = jraph.pad_with_graphs(graph, n_node=8, n_edge=8)

        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        config = get_class_config()
        config.use_layer_norm = False
        config.use_batch_norm = False
        net_fn = MPEU(config, True)
        net = hk.with_empty_state(hk.transform(net_fn))
        # Create weights for the model
        params, state = net.init(init_rng, init_graphs)

        graphs_pred = net.apply(params, state, rng, graphs)
        self.assertEqual(len(graphs_pred), 2)
        prediction = graphs_pred[0].globals
        self.assertEqual(np.shape(prediction)[-1], 2)

    def test_edge_update_fn(self):
        """Test the expected edge and message feature vector in the edge update.

        Pass a single edge into the edge update function and compare with
        expected output.
        """
        latent_size = 10
        hk_init = hk.initializers.Identity()
        edge_update_fn = _get_edge_update_fn(
            latent_size, hk_init, use_layer_norm=False, use_batch_norm=False,
            activation=shifted_softplus, dropout_rate=0, mlp_depth=2, is_training=False)

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
        # Initialize weights and paramters with haiku.
        # We use a different to initialize, this tests the ability of the
        # function to use different inputs.
        params = hk_edge_update_fn.init(
            init_rng,
            {'edges': jnp.zeros((2)), 'message': jnp.zeros(15)},
            jnp.zeros((4)), jnp.zeros((4)),
            jnp.zeros((11)))

        edge_message_update = hk_edge_update_fn.apply(
            params, rng, edge_message, sent_attributes, received_attributes,
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
        fun = _get_edge_embedding_fn(latent_size, k_max, delta, mu_min)
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
        node_embedding_fn = _get_node_embedding_fn(
            latent_size, max_atomic_number, False, False,
            shifted_softplus, hk_init, True)
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
        node_update_fn = _get_node_update_fn(
            latent_size, hk_init, use_layer_norm=False, use_batch_norm=False,
            activation=shifted_softplus, dropout_rate=0.0, mlp_depth=2,
            is_training=True)

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
        params = node_update_fn.init(
            init_rng,
            nodes, sent_message, received_message, global_attributes
        )

        nodes_updated = node_update_fn.apply(
            params, rng,
            nodes, sent_message, received_message, global_attributes
        )
        np.testing.assert_allclose(
            nodes_updated,
            shifted_softplus(received_message['messages']) + nodes)

    def test_readout_function(self):
        """Test the output of the readout function against the expected result.
        """

        latent_size = 16
        hk_init = hk.initializers.Identity()
        readout_node_update_fn = _get_readout_node_update_fn(
            latent_size, hk_init, use_layer_norm=False, use_batch_norm=False,
            dropout_rate=0.0, activation=shifted_softplus, output_size=1,
            is_training=True)
        # TODO: finish test


if __name__ == '__main__':
    unittest.main()

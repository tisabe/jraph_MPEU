"""Test our GNN Models.


Right now we have a test with 0/1s on a graph but we only use self edges
since otherwise the complexity means it's hard to figure out what the right
answer is on paper to check we're getting the right answer.
"""
import unittest
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import optax

import ml_collections

from models import (
    GNN,
    Set2Set,
    shifted_softplus,
    get_edge_update_fn,
    get_edge_embedding_fn,
    get_node_embedding_fn,
    get_node_update_fn,
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
            edges=jnp.ones((n_edge)),
            senders=jnp.array([0, 0, 1, 2]),
            receivers=jnp.array([1, 2, 0, 0]),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_edge]),
            globals=None
        )

        n_node = jnp.array([3])
        n_edge = jnp.array([4])
        node_features = jnp.array([0, 1, 2])
        edge_features = jnp.array([1, 2, 3, 4])
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

        net_fn = GNN(self.config)
        net = hk.without_apply_rng(hk.transform(net_fn))
        # Create weights for the model
        params = net.init(init_rng, init_graphs)

        graph_pred = net.apply(params, graph)
        prediction = graph_pred.globals

        # Here, we calculate the expected result, starting with the embeddings.
        nodes_embedded = jnp.ones((int(n_node), latent_size))/latent_size
        edge_embedding_fn = get_edge_embedding_fn(
            latent_size,
            self.config.k_max,
            self.config.delta,
            self.config.mu_min)
        edges_embedded = edge_embedding_fn(edge_features)['edges']
        edges_embedded = np.array(edges_embedded)

        # The updated edges.
        edge_factor = (np.sum(edges_embedded, axis=-1) + 2)/latent_size
        edge_factor = shifted_softplus(edge_factor)*2
        edge_factor = np.array(edge_factor)

        edge_updated = jnp.ones((int(n_edge), latent_size))
        for i in range(int(n_edge)):
            edge_updated = edge_updated.at[i].set(
                edge_updated[i] * float(edge_factor[i]))

        # The edge-wise message.
        messages = shifted_softplus(edge_updated)
        messages = shifted_softplus(messages)/latent_size

        # Node-wise message
        message_node = jnp.zeros((int(n_node), latent_size))
        message_node = message_node.at[0].set(messages[1])
        message_node = message_node.at[1].set(messages[0] + messages[2])
        message_node = message_node.at[2].set(messages[3])

        # State-transition
        nodes_updated = nodes_embedded + shifted_softplus(message_node)

        # Readout
        nodes_readout = shifted_softplus(nodes_updated[:, 0])/2
        prediction_expected = jnp.sum(nodes_readout)

        np.testing.assert_array_equal(edge_updated, graph_pred.edges['edges'])
        self.assertEqual(prediction_expected, prediction)


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

    def test_set2set(self):
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        latent_size = 64
        num_passes = 2
        batch_size = 2
        set2set = Set2Set(latent_size, num_passes, batch_size)
        net = hk.without_apply_rng(hk.transform(set2set))

        num_nodes = 10
        segment_ids = jnp.array([0,0,0,0,0,1,1,1,1,1])

        data = jnp.ones((num_nodes, latent_size))
        params = net.init(init_rng, data, segment_ids)

        #print(params)
        pred = net.apply(params, data, segment_ids)
        #print(pred)

    def test_set2set_integration(self):
        """Test integrating set2set into a GraphNetwork. """
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        latent_size = 64
        num_passes = 2
        batch_size = 2
        set2set = Set2Set(latent_size, num_passes, batch_size)

        g_update = lambda nodes, edges, globals: nodes
        net_fn = jraph.GraphNetwork(
                update_edge_fn=None,
                update_node_fn=None,
                update_global_fn=g_update,
                aggregate_nodes_for_globals_fn=set2set)
        n_node = 10
        n_edge = 4
        test_graph = jraph.GraphsTuple(
            nodes=jnp.ones((n_node)),
            edges=jnp.ones((n_edge)),
            senders=jnp.array([0,0,1,2]),
            receivers=jnp.array([1,2,0,0]),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_edge]),
            globals=None
        )
        #print(test_graph)
        graphs = jraph.batch([test_graph]*batch_size)

        net = hk.without_apply_rng(hk.transform(net_fn))
        params = net.init(init_rng, graphs) # create weights etc. for the model
        
        graph_pred = net.apply(params, graphs)
        prediction = graph_pred.globals

        # test jitted function
        net_apply_jit = jax.jit(net.apply)
        graph_pred = net_apply_jit(params, graphs)
        prediction = graph_pred.globals
        print(type(prediction))

    def test_set2set_fit(self):
        """Test the set2set function by generating artificial sets and training
        on properties of these sets"""
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        latent_size = 64
        num_passes = 2
        batch_size = 32
        num_nodes = 256
        output_size = 1
        # define the network
        
        class Module:
            def __init__(self, latent_size, num_passes, batch_size, output_size):
                self.latent_size = latent_size
                self.num_passes = num_passes
                self.batch_size = batch_size
                self.output_size = output_size

            def __call__(self, 
                data: jnp.ndarray,
                segment_ids: jnp.ndarray,
                num_segments: int
            ):
                set2set = Set2Set(self.latent_size, self.num_passes, 
                        self.batch_size)
                linear = hk.Linear(self.output_size)

                qstar = set2set(data, segment_ids, num_segments=num_segments, 
                        indices_are_sorted=None,
                        unique_indices=None)
                out = linear(qstar)
                return out

        rng, data_rng = jax.random.split(rng)
        data = jax.random.normal(data_rng, (num_nodes, 1))
        segment_ids = jnp.linspace(0, batch_size-1, num_nodes, dtype='int32')

        net_fn = Module(latent_size, num_passes, batch_size, output_size)
        net = hk.without_apply_rng(hk.transform(net_fn))
        params = net.init(init_rng, data, segment_ids, num_segments=batch_size
        ) # create weights etc. for the model

        step_size = 1e-4
        opt_init, opt_update = optax.adam(step_size)
        opt_state = opt_init(params)

        def get_random_segment_ids(N, batch_size, key):
            assert(N >= batch_size)
            segment_ids = jnp.arange(0, batch_size, dtype='int32')
            num_ids_left = N - batch_size
            ids_add = jax.random.uniform(key, (num_ids_left,))*batch_size
            ids_add = jnp.array(ids_add, dtype='int32')
            return jnp.concatenate([segment_ids, ids_add], axis=0)
        
        @partial(jax.jit, static_argnums=(4,))
        def train_step(params, opt_state, data, segment_ids, num_segments):
            def loss_fn(params, data, segment_ids):
                pred = net.apply(params, data, segment_ids, num_segments)
                target = jax.ops.segment_max(data, segment_ids, num_segments)
                return jnp.mean(jnp.abs(pred - target))
            # compute gradient function and gradients
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(params, data, segment_ids)
            # update the params with gradient
            updates, opt_state = opt_update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state, loss

        num_steps = 10000
        for i in range(num_steps):
            # generate new data at every step
            rng, data_rng = jax.random.split(rng)
            data = jax.random.normal(data_rng, (num_nodes, 1))
            segment_ids = get_random_segment_ids(num_nodes, batch_size, data_rng)

            params, opt_state, loss = train_step(
                    params, opt_state, data, segment_ids, batch_size)
            if i%1000 == 0:
                print(loss)

        # generate a final example
        rng, data_rng = jax.random.split(rng)
        data = jax.random.normal(data_rng, (num_nodes, 1))
        segment_ids = get_random_segment_ids(num_nodes, batch_size, data_rng)
        
        pred = net.apply(params, data, segment_ids, batch_size)
        print(f'targets: {jax.ops.segment_max(data, segment_ids)[:,0]}')
        print(f'prediction: {pred[:,0]}')

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

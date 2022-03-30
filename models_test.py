import numpy as np
import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import optax

from absl import flags
import ml_collections
from ml_collections import config_flags

import unittest
from functools import partial

from models import (
    GNN, 
    shifted_softplus, 
    get_edge_update_fn, 
    get_edge_embedding_fn,
    get_node_embedding_fn,
    Set2Set
)


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
        
        self.config.latent_size = 64
        self.config.message_passing_steps = 3
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

    
    def test_edge_update_fn(self):
        latent_size = 10
        hk_init = hk.initializers.Identity()
        edge_update_fn = get_edge_update_fn(latent_size, hk_init)
        
        sent_attributes = jnp.arange(0,4) 
        received_attributes = jnp.arange(4,8)
        edge = jnp.arange(8,10)
        message = jnp.arange(0,15)
        edge_message = {'edges': edge, 'messages': message}
        graph_globals = jnp.ones((11))

        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        fun = hk.transform(edge_update_fn)
        fun = hk.without_apply_rng(fun)
        params = fun.init(init_rng, 
            {'edges': jnp.zeros((2)), 'message': jnp.zeros(15)}, 
            jnp.zeros((4)), jnp.zeros((4)), 
            jnp.zeros((11)))

        edge_message_update = fun.apply(params, 
            edge_message, sent_attributes, received_attributes, 
            graph_globals)

        edge_updated = edge_message_update['edges']
        message_updated = edge_message_update['messages']

        edge_expected = shifted_softplus(jnp.arange(0,10))
        np.testing.assert_allclose(edge_updated, edge_expected)
        
        # test the result of message calculation
        node_message_expected = jnp.pad(sent_attributes, (0,6))
        edge_message_expected = shifted_softplus(shifted_softplus(edge_expected))
        message_expected = jnp.multiply(node_message_expected, edge_message_expected)

        np.testing.assert_allclose(message_updated, message_expected)
        return 0

    def test_edge_embedding_fn(self):
        latent_size = 64
        edges = jnp.ones((1))
        k_max = 150
        k_vec = jnp.arange(0, k_max)
        delta = 0.1
        mu_min = 0.2
        fun = get_edge_embedding_fn(latent_size, k_max, delta, mu_min)
        embedding = fun(edges)

        embedding_expected = [jnp.exp(-jnp.square(edges[0]-(-mu_min+k_vec*delta))/delta)]

        np.testing.assert_allclose(embedding['edges'], embedding_expected, atol=6)
        np.testing.assert_array_equal(embedding['messages'], jnp.zeros((1,latent_size)))

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
        
        np.testing.assert_array_equal([len(nodes),latent_size],
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
        #print(prediction)

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

        num_steps = 100000
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

if __name__ == '__main__':
    unittest.main()
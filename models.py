import jax
import jax.numpy as jnp
import jraph
import numpy as np
import haiku as hk
import ml_collections

from typing import (
    Optional
)


# define the shifted softplus activation function
LOG2 = jnp.log(2)
def shifted_softplus(x: jnp.ndarray) -> jnp.ndarray:
    # continuous version of relu activation function
    return jnp.logaddexp(x, 0) - LOG2


def get_set2set_fn(init_graph, latent_size, num_passes):
    '''Return the set2set function. It has the same call signature as 
    jraph.segment_sum, so it can be used as aggregation function 
    in a jraph.GraphNetwork.'''
    lstm = hk.LSTM((batch_size, latent_size))
    q_star = jnp.zeros((batch_size, latent_size))

    lstm_init = lstm.initial_state(batch_size)
    next_state = lstm_init

    def set2set(data: jnp.ndarray,
        segment_ids: jnp.ndarray,
        num_segments: Optional[int] = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False
    ):
        m_i = data
        for i in range(num_passes):
            q, next_state = lstm(q_star, next_state)
            e_i = jnp.sum(jnp.multiply(m_i, q), axis=-1)
            a_i = jraph.segment_softmax(e_i, 
                segment_ids, num_segments, 
                indices_are_sorted, unique_indices)
            r = jraph.segment_sum(
                jnp.multiply(a_i, m_i), 
                segment_ids, num_segments, 
                indices_are_sorted, unique_indices)
            q_star = jnp.concatenate((q, r), axis=-1)

        return q_star
    return set2set


def get_edge_embedding_fn(latent_size, k_max, delta, mu_min):
    '''Return the edge embedding function'''
    def edge_embedding_fn(edges) -> jnp.ndarray:
        '''Edge embedding function for general data.'''
        distances = edges

        k_vals = jnp.arange(0, k_max)
        # create tiled matrices with distances and k values
        k_mesh, d_mesh = jnp.meshgrid(k_vals, distances)

        exponents = -1 * jnp.square(d_mesh + mu_min - k_mesh * delta) / delta
        edges = jnp.exp(exponents)

        # now we define a structured vector, to combine edge and message features
        edge_message = {'edges': edges, 
            'messages': jnp.zeros((edges.shape[0], latent_size))}
        return edge_message
    return edge_embedding_fn


def get_node_embedding_fn(latent_size, max_atomic_number, hk_init=None):
    '''Return the node embedding function.'''
    def node_embedding_fn(nodes) -> jnp.ndarray:
        '''Embedds the node features using one-hot encoding and a dense nn layer.'''
        net = hk.Linear(latent_size, with_bias=False, w_init=hk_init)
        nodes = jax.nn.one_hot(nodes, max_atomic_number)
        return net(nodes)
    return node_embedding_fn


def get_embedder(config: ml_collections.ConfigDict):
    '''Return the embedding function, with hyperparameters defined in config.'''
    latent_size = config.latent_size
    k_max = config.k_max
    delta = config.delta
    mu_min = config.mu_min
    max_atomic_number = config.max_atomic_number
    hk_init = config.hk_init

    embedder = jraph.GraphMapFeatures(
        embed_edge_fn=get_edge_embedding_fn(latent_size, k_max, delta, mu_min),
        embed_node_fn=get_node_embedding_fn(latent_size, max_atomic_number, hk_init),
        embed_global_fn=None
    )
    return embedder


def get_edge_update_fn(latent_size, hk_init):
    '''Return the edge update function.'''
    def edge_update_fn(edge_message, sent_attributes, received_attributes,
                    global_edge_attributes) -> jnp.ndarray:
        #return edge_message
        '''Edge update and message function for graph net.'''
        # first, compute edge update
        edge_node_concat = jnp.concatenate([sent_attributes, received_attributes, 
            edge_message['edges']], axis=-1)
        
        net_edge = hk.Sequential(
            [hk.Linear(2 * latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init)])
        # TODO: check with PBJ which activation function is at end of edge update
        
        edge_message['edges'] = net_edge(edge_node_concat)

        # then, compute edge-wise messages
        net_message_edge = hk.Sequential(
            [hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus])

        net_message_node = hk.Linear(latent_size, with_bias=False, w_init=hk_init)

        edge_message['messages'] = jnp.multiply(net_message_edge(edge_message['edges']),
                                                net_message_node(sent_attributes))
        return edge_message
    return edge_update_fn


def get_node_update_fn(latent_size, hk_init):
    '''Return the node update function.'''
    def node_update_fn(nodes, sent_attributes,
                   received_attributes, global_attributes) -> jnp.ndarray:
        '''Node update function for graph net.'''
        net = hk.Sequential(
            [hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init)])

        messages_propagated = net(received_attributes['messages'])

        return nodes + messages_propagated
    return node_update_fn


def get_readout_global_fn():
    '''Return the readout global function.'''
    def readout_global_fn(node_attributes, edge_attributes, globals_) -> jnp.ndarray:
        '''Global readout function for graph net.
        Returns the aggregated node features, using the aggregation function 
        defined in config'''
        return node_attributes
    return readout_global_fn


def get_readout_node_update_fn(latent_size, hk_init):
    '''Return readout node update function.'''
    def readout_node_update_fn(nodes, sent_attributes,
                           received_attributes, global_attributes) -> jnp.ndarray:
        '''Node update function for readout phase in graph net.
        
        Args:
            nodes: jnp.ndarray, node features to be updated
            sent_attributes: aggregated sender features, not used
            received_attributes: aggregated receiver features, not used
            global_attributs: global features, not used

        Returns:
            jnp.ndarray, updated node features
        '''
        net = hk.Sequential(
            [hk.Linear(latent_size // 2, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(1, with_bias=False, w_init=hk_init)])
        return net(nodes)
    return readout_node_update_fn


class GNN:

    def __init__(self, config: ml_collections.ConfigDict):
        self.config = config

        if self.config.aggregation_message_type=='sum':
            self.aggregation_message_fn = jraph.segment_sum
        elif self.config.aggregation_message_type=='mean':
            self.aggregation_message_fn = jraph.segment_mean
        else:
            raise ValueError(f'Aggregation type {self.config.aggregation_message_type} not recognized')

        if self.config.aggregation_readout_type=='sum':
            self.aggregation_readout_fn = jraph.segment_sum
        elif self.config.aggregation_readout_type=='mean':
            self.aggregation_readout_fn = jraph.segment_mean
        else:
            raise ValueError(f'Aggregation type {self.config.aggregation_readout_type} not recognized')
    
    
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:

        # Add a global paramater for graph classification. It has shape (len(n_node), 1))
        graphs = graphs._replace(globals=jnp.zeros([graphs.n_node.shape[0], 1], dtype=np.float32))

        embedder = get_embedder(self.config)
        # embed the graph with embedder
        graphs = embedder(graphs)

        for _ in range(self.config.message_passing_steps):
            net = jraph.GraphNetwork(
                update_node_fn=get_node_update_fn(self.config.latent_size, self.config.hk_init),
                update_edge_fn=get_edge_update_fn(self.config.latent_size, self.config.hk_init),
                update_global_fn=None,
                aggregate_edges_for_nodes_fn=self.aggregation_message_fn)
            graphs = net(graphs)

        net_readout = jraph.GraphNetwork(
            update_node_fn=get_readout_node_update_fn(self.config.latent_size, self.config.hk_init),
            update_edge_fn=None,
            update_global_fn=get_readout_global_fn(),
            aggregate_nodes_for_globals_fn=self.aggregation_readout_fn)
        # apply readout function on graph
        graphs = net_readout(graphs)

        return graphs
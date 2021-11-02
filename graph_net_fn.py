import jax
import jax.numpy as jnp
import jraph
from jraph._src import utils
import numpy as np
import haiku as hk
import config

from utils import *


def edge_embedding_fn(edges, sent_attributes, received_attributes,
                      global_edge_attributes) -> jnp.ndarray:
    """Edge embedding function for general data."""
    distances = edges

    k_max = 150  # TODO: make parameters accessible
    delta = 0.1
    mu_min = 0.0

    k_vals = jnp.arange(0, k_max)
    # create tiled matrices with distances and k values
    k_mesh, d_mesh = jnp.meshgrid(k_vals, distances)

    exponents = -1 * jnp.square(d_mesh + mu_min - k_mesh * delta) / delta
    edges = jnp.exp(exponents)

    # now we define a structured vector, to combine edge and message features
    edge_message = {'edges': edges, 'messages': jnp.zeros((edges.shape[0], config.N_HIDDEN_C))}
    return edge_message


def node_embedding_fn(nodes, sent_attributes,
                      received_attributes, global_attributes) -> jnp.ndarray:
    """Node embedding function for aflow data."""
    # TODO: look up how it is implemented in MPEU
    net = hk.Linear(config.N_HIDDEN_C, with_bias=False)
    nodes = jax.nn.one_hot(nodes, config.MAX_ATOMIC_NUMBER)
    return net(nodes)


def net_embedding():
    net = jraph.GraphNetwork(
        update_node_fn=node_embedding_fn,
        update_edge_fn=edge_embedding_fn,
        update_global_fn=None)
    return net


# define the shifted softplus activation function
LOG2 = jnp.log(2)


def shifted_softplus(x: jnp.ndarray) -> jnp.ndarray:
    # continuous version of relu activation function
    return jnp.logaddexp(x, 0) - LOG2


def edge_update_fn(edge_message, sent_attributes, received_attributes,
                   global_edge_attributes) -> jnp.ndarray:
    """Edge update and message function for graph net."""
    # first, compute edge update
    edge_node_concat = jnp.concatenate([edge_message['edges'], sent_attributes, received_attributes], axis=-1)
    #print(jnp.shape(edge_node_concat))
    net_e = hk.Sequential(
        [hk.Linear(2 * config.N_HIDDEN_C, with_bias=False),
         shifted_softplus,
         hk.Linear(config.N_HIDDEN_C, with_bias=False)])
    edge_message['edges'] = net_e(edge_node_concat)

    # then, compute edge-wise messages
    net_m_e = hk.Sequential(
        [hk.Linear(config.N_HIDDEN_C, with_bias=False),
         shifted_softplus,
         hk.Linear(config.N_HIDDEN_C, with_bias=False),
         shifted_softplus])

    net_m_n = hk.Linear(config.N_HIDDEN_C, with_bias=False)

    edge_message['messages'] = jnp.multiply(net_m_e(edge_message['edges']),
                                            net_m_n(received_attributes))
    return edge_message


def node_update_fn(nodes, sent_attributes,
                   received_attributes, global_attributes) -> jnp.ndarray:
    """Node update function for graph net."""
    net = hk.Sequential(
        [hk.Linear(config.N_HIDDEN_C, with_bias=False),
         shifted_softplus,
         hk.Linear(config.N_HIDDEN_C, with_bias=False)])

    messages_propagated = net(received_attributes['messages'])

    return nodes + messages_propagated


def readout_global_fn(node_attributes, edge_attributes, globals_) -> jnp.ndarray:
    """Global readout function for graph net."""
    # return the aggregated node features
    return node_attributes


def readout_node_update_fn(nodes, sent_attributes,
                           received_attributes, global_attributes) -> jnp.ndarray:
    """Node readout function for graph net."""
    net = hk.Sequential(
        [hk.Linear(config.N_HIDDEN_C // 2, with_bias=False),
         shifted_softplus,
         hk.Linear(config.LABEL_SIZE, with_bias=False)])

    return net(nodes)


def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # Add a global paramater for graph classification. It has shape (len(n_node, LABEL_SIZE))
    graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], config.LABEL_SIZE], dtype=np.float32))

    if config.AVG_MESSAGE:
        aggregate_edges_for_nodes_fn = jraph.segment_mean
    else:
        aggregate_edges_for_nodes_fn = utils.segment_sum

    if config.AVG_READOUT:
        aggregate_nodes_for_globals_fn = jraph.segment_mean
    else:
        aggregate_nodes_for_globals_fn = utils.segment_sum

    embedder = net_embedding()
    net = jraph.GraphNetwork(
        update_node_fn=node_update_fn,
        update_edge_fn=edge_update_fn,
        update_global_fn=None,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)
    net_2 = jraph.GraphNetwork(
        update_node_fn=node_update_fn,
        update_edge_fn=edge_update_fn,
        update_global_fn=None,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)
    net_3 = jraph.GraphNetwork(
        update_node_fn=node_update_fn,
        update_edge_fn=edge_update_fn,
        update_global_fn=None,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)
    net_readout = jraph.GraphNetwork(
        update_node_fn=readout_node_update_fn,
        update_edge_fn=None,
        update_global_fn=readout_global_fn,
        aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn)
    # return net(embedder(graph))
    graph = embedder(graph)
    graph = net(graph)
    graph = net_2(graph)
    graph = net_3(graph)
    graph = net_readout(graph)
    return graph

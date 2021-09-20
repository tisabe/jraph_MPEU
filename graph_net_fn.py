import jax
import jax.numpy as jnp
import spektral
import jraph
import numpy as np
import haiku as hk

from utils import *

# define some hidden layer dimensions
N_HIDDEN_C = 32  # C from MPEU paper for hidden layers

# to get label size, we need an example dataset
dataset = QM9(amount=1)
graph_j, label = spektral_to_jraph(dataset[0])
LABEL_SIZE = len(label)


def edge_embedding_fn(edges, sent_attributes, received_attributes,
                      global_edge_attributes) -> jnp.ndarray:
    """Edge embedding function for QM9 data."""

    # we extract the coordinates from the node attributes
    sender_xyz = sent_attributes[:, 5:8]
    receiver_xyz = received_attributes[:, 5:8]

    distances_vec = sender_xyz - receiver_xyz
    distances = jnp.linalg.norm(distances_vec, ord=2, axis=-1)

    k_max = 150  # TODO: make parameters accessible
    delta = 0.1
    mu_min = 0.0

    k_vals = jnp.arange(0, k_max)
    # create tiled matrices with distances and k values
    k_mesh, d_mesh = jnp.meshgrid(k_vals, distances)

    exponents = -1 * jnp.square(d_mesh + mu_min - k_mesh * delta) / delta
    edges = jnp.exp(exponents)

    # now we define a structured vector, to combine edge and message features
    edge_message = {'edges': edges, 'messages': jnp.zeros((edges.shape[0], N_HIDDEN_C))}
    return edge_message


def node_embedding_fn(nodes, sent_attributes,
                      received_attributes, global_attributes) -> jnp.ndarray:
    """Node embedding function for QM9 data."""
    # TODO: look up how it is implemented in MPEU
    net = hk.Linear(N_HIDDEN_C, with_bias=False)
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
    net_e = hk.Sequential(
        [hk.Linear(2 * N_HIDDEN_C, with_bias=False),
         shifted_softplus,
         hk.Linear(N_HIDDEN_C, with_bias=False)])
    edge_message['edges'] = net_e(edge_node_concat)

    # then, compute edge-wise messages
    net_m_e = hk.Sequential(
        [hk.Linear(N_HIDDEN_C, with_bias=False),
         shifted_softplus,
         hk.Linear(N_HIDDEN_C, with_bias=False),
         shifted_softplus])

    net_m_n = hk.Linear(N_HIDDEN_C, with_bias=False)

    edge_message['messages'] = jnp.multiply(net_m_e(edge_message['edges']),
                                            net_m_n(received_attributes))
    return edge_message


def node_update_fn(nodes, sent_attributes,
                   received_attributes, global_attributes) -> jnp.ndarray:
    """Node update function for graph net."""
    net = hk.Sequential(
        [hk.Linear(N_HIDDEN_C, with_bias=False),
         shifted_softplus,
         hk.Linear(N_HIDDEN_C, with_bias=False)])

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
        [hk.Linear(N_HIDDEN_C // 2, with_bias=False),
         shifted_softplus,
         hk.Linear(LABEL_SIZE, with_bias=False)])

    return net(nodes)


def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # Add a global paramater for graph classification. It has shape (len(n_node, LABEL_SIZE))
    graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], LABEL_SIZE], dtype=np.float32))

    embedder = net_embedding()
    net = jraph.GraphNetwork(
        update_node_fn=node_update_fn,
        update_edge_fn=edge_update_fn,
        update_global_fn=None)
    net_2 = jraph.GraphNetwork(
        update_node_fn=node_update_fn,
        update_edge_fn=edge_update_fn,
        update_global_fn=None)
    net_3 = jraph.GraphNetwork(
        update_node_fn=node_update_fn,
        update_edge_fn=edge_update_fn,
        update_global_fn=None)
    net_readout = jraph.GraphNetwork(
        update_node_fn=readout_node_update_fn,
        update_edge_fn=None,
        update_global_fn=readout_global_fn)
    # return net(embedder(graph))
    graph = embedder(graph)
    graph = net(graph)
    graph = net_2(graph)
    graph = net_3(graph)
    graph = net_readout(graph)
    return graph

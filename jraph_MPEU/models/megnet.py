"""Alternative models to compare the Message passing with edge updates with.

First the specific update functions are defined and then they are composed into
a jraph graph network.
"""

import jax.numpy as jnp
import jraph
import numpy as np
import haiku as hk
import ml_collections

from jraph_MPEU.models import shifted_softplus
from jraph_MPEU.models.mpeu import get_node_embedding_fn


def get_edge_embedding_MEG(
        latent_size: int, k_max: int, delta: float, mu_min: float, hk_init=None):
    """Return the edge embedding function, analogous to MEGNet.

    The distance between nodes gets passed into radial basis functions to
    get initial edge vector embeddings (known as eps_vw in the paper). The
    radial basis functions have parameter k that range from k=0 to k_max. The
    functions are all shifted by mu_min. The k_max parameter determines the
    vector size of the initial edge embedding.

    Args:
        latent_size: Size of the node feature and edge feature vectors. This
            also the size of the node feature embedding. In the PB Jorgensen
            paper this is C.
        k_max: Size of the RBF expansion to get the initial edge vectors which
            are labelled eps_vw in the paper.
        delta: Spread of the RBF functions from one value of k to k+1. Somewhat
            close to the standard deviation of a gaussian. See equation (5) in
            the PB Jorgensen paper for more details.
        mu_min: Constant shift for the RBF expansion basis functions.
        hk_init: Linear weight intializer function for our linear embedding.
    """
    def edge_embedding_fn(edges) -> jnp.ndarray:
        """Edge embedding function for general data."""
        distances = edges
        k_vals = jnp.arange(0, k_max)
        # Create tiled matrices with distances and k values.
        k_mesh, d_mesh = jnp.meshgrid(k_vals, distances)
        # Exponents is the argument fed into the exp function a line further
        # down.
        exponents = -1 * jnp.square(d_mesh + mu_min - k_mesh * delta) / delta
        edges = jnp.exp(exponents)

        net = hk.Linear(latent_size, with_bias=False, w_init=hk_init)

        return net(edges)
    return edge_embedding_fn


def get_embedder(config: ml_collections.ConfigDict):
    """Return embedding function, with hyperparameters defined in config.

    This is a wrapper function that maps the embedding function to our graphs
    using jraph.
    """
    # Set constant values based on config.
    latent_size = config.latent_size
    k_max = config.k_max
    delta = config.delta
    mu_min = config.mu_min
    max_atomic_number = config.max_atomic_number
    hk_init = config.hk_init
    # Map embedding function and node embedding function to graphs using
    # jraph.
    embedder = jraph.GraphMapFeatures(
        embed_edge_fn=get_edge_embedding_MEG(
            latent_size, k_max, delta, mu_min, hk_init),
        embed_node_fn=get_node_embedding_fn(
            latent_size, max_atomic_number, hk_init),
        embed_global_fn=None)
    return embedder


def get_node_update_MEG(latent_size, hk_init):
    """Return the node update function.

    Wrapper function so that we can interface with jraph.

    Args:
        latent_size: The size of the node feature vector and message vector.
        hk_init: The weight intializer function for the node feature vector
            update network.
    """
    def node_update_fn(
            nodes, sent_attributes,
            received_attributes, global_attributes) -> jnp.ndarray:
        """Node update function for MEG block.

        Args:
            nodes: Node feature vector at previous iteration (h_i(t-1)).
            sent_attributes: Not used, but expected by jraph.
            received_attributes: The aggregation of incoming edge features.
        """
        # These input arguments are not used but expected by jraph.
        del sent_attributes, global_attributes
        net = hk.Sequential([
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init)])

        # concatenate node- and received features
        features = jnp.concatenate([nodes, received_attributes], axis=-1)
        return net(features)
    return node_update_fn


def get_edge_update_MEG(latent_size, hk_init):
    """Return edge update function for MEGNet-like network.

    Wrapper function so we can interface with jraph.

    Args:
        latent_size: The size of the node feature vector and message vector.
        hk_init: The weight intializer function for the node feature vector
            update network.
    """
    def edge_update_fn(
            edges, sent_attributes, received_attributes,
            global_edge_attributes) -> jnp.ndarray:
        """Edge update and message function for graph net.

        Args:
            edge_message: dictionary containing two keys, the edge vector
                and the message corresponding to that edge. Both of size
                latent_size (size C in the paper).
            sent_attributes: Node feature vector from sending node to be
                concatenated with the edge vector.
            receive_attributes: Node feature vector from the receiving node to
                be concatenated with the edge vector.
            global_edge_attributes: Not used but expected by jraph function.
        """
        del global_edge_attributes  # Not used but expected by jraph.

        # Concatenate the sending node feature vector with the receiving node
        # feature vector and the edge vector for the edge connecting the
        # sending and receiving node.
        features = jnp.concatenate([
            sent_attributes, received_attributes,
            edges], axis=-1)
        net = hk.Sequential([
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init)])
        return net(features)
    return edge_update_fn


def get_block_update_MEG(latent_size, hk_init):
    """Return the MEGNet block as a callable function with a graph as input.

    Args:
        latent_size: The size of the node feature vector and message vector.
        hk_init: The weight intializer function for the node feature vector
            update network.
    """
    # define a network for preprocessing the graph, with a two layer neural net
    net_preprocess = jraph.GraphMapFeatures(
        hk.Sequential([
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init)]),
        hk.Sequential([
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init)]),
        hk.Sequential([
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init)]),
    )
    net_block = jraph.GraphNetwork(
        update_node_fn=get_node_update_MEG(
            latent_size, hk_init),
        update_edge_fn=get_edge_update_MEG(
            latent_size, hk_init),
        update_global_fn=None,
        aggregate_edges_for_nodes_fn=jraph.segment_mean)

    def update_block_MEG(graphs):
        """Return updated jraph.GraphsTuple"""
        nodes = graphs.nodes[:]
        edges = graphs.edges[:]

        graphs_preprocessed = net_preprocess(graphs)
        # save for skip connection
        nodes_preprocessed = graphs_preprocessed.nodes[:]
        edges_preprocessed = graphs_preprocessed.edges[:]

        graphs_updated = net_block(graphs_preprocessed)
        graphs.nodes = nodes + nodes_preprocessed + graphs_updated.nodes
        graphs.edges = edges + edges_preprocessed + graphs_updated.edges

        return graphs
    return update_block_MEG


def get_readout_global_fn():
    """Return the readout global function."""
    def readout_global_fn(
            node_attributes, edge_attributes,
            global_attributes) -> jnp.ndarray:
        """Global readout function for graph net.

        Returns the aggregated node features, using the aggregation function
        defined in config.

        Args:
            node_attributes: Aggregated node feature vectors.
            edge_attributes: Not used but expected by jraph.
            global_attributes: Not used but expected by jraph.
        """
        # Delete unused arguments that are expected by jraph.
        del edge_attributes, global_attributes
        return node_attributes
    return readout_global_fn


def get_readout_node_update_fn(latent_size, hk_init):
    """Return readout node update function.

    This is a wrapper function for the readout_node_update_fn.

    Args:
        latent_size: The node feature vector dimensionality.
        hk_init: The weight initializer for the the readout NN.
    """
    def readout_node_update_fn(
            nodes, sent_attributes,
            received_attributes, global_attributes
    ) -> jnp.ndarray:
        """Node update function for readout phase in graph net.

        Note: In future iterations we might want to include the
            sent/received attributes and some global attributes.

        Args:
            nodes: jnp.ndarray, array of node features to be updated.
            sent_attributes: aggregated sender features, not used.
            received_attributes: aggregated receiver features, not used.
            global_attributes: global features, not used.

        Returns:
            jnp.ndarray, updated node features
        """
        del sent_attributes, received_attributes, global_attributes

        # NN. First layer has output size of latent_size/2 (C/2) with
        # shifted softplus. Then we have a linear FC layer with no softplus.
        net = hk.Sequential([
            hk.Linear(latent_size // 2, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(1, with_bias=False, w_init=hk_init)])
        # Apply the network to the nodes.
        return net(nodes)
    return readout_node_update_fn


class MEGNet:
    """Graph neural network class where we define the interactions/updates."""
    def __init__(self, config: ml_collections.ConfigDict):
        """Initialize the GNN using a config.

        Args:
            config: Configuration specifying GNN properties.
        """
        self.config = config

    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Call function to do forward pass of the GNN.

        Args:
            graphs: The input graphs on which to do the forward pass.

        Returns:
            Returns the same graphs as the input.
        """
        # Add a global paramater for graph classification.
        # graphs.n_node.shape[0] is the batch dimension. We intiialize the
        # globals of the graph to zero, we don't use this, but we could in
        # the future if we wanted to give each graph a global input label.
        # We leave this as zero as a placeholder for future work streams that
        # use global labels. We don't put the graph output labels here since
        # we don't want to carry around the right answer with our input to
        # the GNNs.
        graphs = graphs._replace(
            globals=jnp.zeros([graphs.n_node.shape[0], 1], dtype=np.float32))

        embedder = get_embedder(self.config)
        # Embed the graph with embedder functions (nodes and edges get
        # embedded).
        graphs = embedder(graphs)
        # Run a for loop over the number of message passing steps.
        for _ in range(self.config.message_passing_steps):
            # In each step, we apply first the edge updates, the message
            # updates and then the node updates and then global updates (which
            # we don't do yet).
            net = get_block_update_MEG(
                self.config.latent_size, self.config.hk_init)
            # Update the graphs by applying our message passing step on graphs.
            graphs = net(graphs)
        # Then after we're done with message passing steps, we intiialize
        # our readout function. aggregate_nodes_for_globals_fn is used to
        # aggregate features to pass to the update_global_fn.
        net_readout = jraph.GraphNetwork(
            update_node_fn=get_readout_node_update_fn(
                self.config.latent_size, self.config.hk_init),
            update_edge_fn=None,
            update_global_fn=get_readout_global_fn(),
            aggregate_nodes_for_globals_fn=jraph.segment_mean)
        # Apply readout function on graph.
        graphs = net_readout(graphs)
        # Right now we are getting global tasks. If we wanted to get node
        # level tasks we could access the node features of the graphs here.
        return graphs

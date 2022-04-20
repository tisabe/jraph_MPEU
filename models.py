"""Graph neural network models.

We use the following conventions for the network.

In the paper the x_1 to x_N we call initial node features.

After embedding we call them the node features at different iterations
and they are labeled in the paper h_1(0) to h_N(0) for the first iteration.

The distances, d_vw are fed into the RBF expansion to get initial edge features
which are labelled in the paper eps_vw. After the first edge update we get
e_vw(0) which are edge features.

In the first edge update we concatenate the two node features h_v(0) and h_w(0)
, which are both size C (latent_size) vectors, with the initial edge feature
eps_vw which is size k_max. The network then outputs an edge vector e_vw(0)
of size C (latent_size).
"""

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import haiku as hk
import ml_collections


# Define the shifted softplus activation function.
# Compute log(2) outside of shifted_softplus so it only needs
# to be computed once.
LOG2 = jnp.log(2)


def shifted_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Continuous and shifted version of relu activation function.

    The function is: log(exp(x) + 1) - log(2)
    This implementation is from PB Jorgensen while SchNet uses
    log(0.5*exp(x)+0.5).

    The benefit compared to a ReLU is that x < 0 you don't have a dead
    zone. And the function is C(n) smooth, infinitely differentiable
    across whole domain.
    """
    return jnp.logaddexp(x, 0) - LOG2


def get_edge_embedding_fn(
        latent_size: int, k_max: int, delta: float, mu_min: float):
    """Return the edge embedding function.

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

        # Now we define a structured vector, to combine edge and message
        # features. Initialize messages to zero of same size latent_size.
        # They will be set in the first message passing step.
        edge_message = {
            'edges': edges,
            'messages': jnp.zeros((edges.shape[0], latent_size))}
        return edge_message
    return edge_embedding_fn


def get_node_embedding_fn(
        latent_size: int, max_atomic_number: int, hk_init=None):
    """Return the node embedding function.

    We take the node atomic number and one hot encode the number. At this point
    the vector is of size max_atomic_number. We then pass it through an
    linear embedding network (no activation function) whose weights need to be
    learned. The final size of the embedding is latent_size (C in the paper).

    TODO(dts): PB Jorgensen didn't give us a concrete answer about the
        embedding network. We use a linear network. We think this is correct
        based on the PB Jorgensen Pytorch code. Question: what does SchNet do?

    Args:
        latent_size: The size of the node feature vectors and node embeddings.
        max_atomic_number: The max atomic number we expect to see from our
            graphs/systems.
        hk_init: Linear weight intializer function for our linear embedding.
    """
    def node_embedding_fn(nodes) -> jnp.ndarray:
        """Embeds the node features using one-hot encoding, linearly.

        Uses a linear dense nn layer.
        """
        net = hk.Linear(latent_size, with_bias=False, w_init=hk_init)
        nodes = jax.nn.one_hot(nodes, max_atomic_number)
        return net(nodes)
    return node_embedding_fn


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
        embed_edge_fn=get_edge_embedding_fn(
            latent_size, k_max, delta, mu_min),
        embed_node_fn=get_node_embedding_fn(
            latent_size, max_atomic_number, hk_init),
        embed_global_fn=None)
    return embedder


def get_edge_update_fn(latent_size: int, hk_init):
    """Return the edge update function and message update function.

    Takes in the previous edge vector value e_vw(t-1) and concatenates with
    the node feature vectors of node v, h_v(t), and node w, h_w(t). Then passes
    this concatenated vector into a FC network with softplus activation, output
    size of 2C. Then passes that output into a a FC network with no activation,
    output size of C. See Figure 1 or Equation 7 of the PB jorgensen paper for
    more details.

    We also compute the message update function. The reason they are in the
    same function is that they have the same input (edge vector and sending
    and receiving node features). Jraph expects the update functions to be
    combined.

    Note: Figure 1 activations and equation 7 has a inconsistency. This
        was brought up to PB Jorgensen. In Figure 1 it's the identity function
        for the second activation. In Equation 7 it's the softplus actiavtion
        again. PB Jorgensen said there is a mistake in Equation 7 and it should
        be the identity function.


    Note: Would be interesting to see what happens if we use the same network
        for each iteration. Right now each iteration of edge updates uses a
        seperate NN. This was experimented in the very deep GNN paper by
        DeepMind.

    Args:
        latent_size: Dimensionality of the edge and node feature vectors.
        hk_init: Weight initializer function of the edge update functon neural
            network.
    """
    def edge_update_fn(
            edge_message, sent_attributes, received_attributes,
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
        edge_node_concat = jnp.concatenate([
            sent_attributes, received_attributes,
            edge_message['edges']], axis=-1)
        # Edge update network. First network is FC input size 3C (or 2C+k_max
        # for first iteration), output size 2C with shifted softplus
        # activation. Second network is FC, input 2C, output C, identity as
        # actiavtion (e.g. linear network).
        net_edge = hk.Sequential([
            hk.Linear(2 * latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init)])

        edge_message['edges'] = net_edge(edge_node_concat)

        # Then, compute edge-wise messages. The input is the edge feature
        # vector. See Figure 1 middle figure for details. The edge feature
        # vector gets passed through two FC layers with shifted softplus.
        # Dimensionality doesn't change at all from input to output here.
        net_message_edge = hk.Sequential([
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus])
        # We also pass the sending/receiving node feature vector through a
        # linear embedding layer (FC with no actiavtion).
        net_message_node = hk.Linear(
            latent_size, with_bias=False, w_init=hk_init)
        # Then element wise multiply together the output of the
        # net_emessage_edge and the output from feeding the sending/receiving
        # node feature vectors to the net_message node.
        edge_message['messages'] = jnp.multiply(
            net_message_edge(edge_message['edges']),
            net_message_node(sent_attributes))
        return edge_message
    return edge_update_fn


def get_node_update_fn(latent_size, hk_init):
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
        """Node update function for graph net.

        Takes the previous node feature vector and the sum of incoming
        messages. Two layer NN on the sum of incoming messages at t+1 and adds
        this to the previous node feature vector at t. This gives us new node
        feature vector for time t+1.

        See Equation (9) in the PB Jorgensen paper.

        TODO: Right now we follow the PB Jorgensen paper and use this function
            with the sum of incoming messages. The Velickovic paper showed
            different aggregation functions for the messages that might
            be of interest.

        Args:
            nodes: Node feature vector at previous iteration (h_i(t-1)).
            sent_attributes: Not used, but expected by jraph.
            received_attributes: The aggregatation of incoming messages.
        """
        # These input arguments are not used but expected by jraph.
        del sent_attributes, global_attributes
        # First layer is FC with shifted_softplus actiavtion. Second layer
        # is linear.
        net = hk.Sequential([
            hk.Linear(latent_size, with_bias=False, w_init=hk_init),
            shifted_softplus,
            hk.Linear(latent_size, with_bias=False, w_init=hk_init)])
        # Get the messages term of Equation 9 which is the net applied to
        # the aggregation of incoming messages to the node.
        messages_propagated = net(received_attributes['messages'])
        # Add the previous value of the node feature vector.
        return nodes + messages_propagated
    return node_update_fn


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


class GNN:
    """Graph neural network class where we define the interactions/updates."""
    def __init__(self, config: ml_collections.ConfigDict):
        """Initialize the GNN using a config.

        Args:
            config: Configuration specifying GNN properties.
        """
        self.config = config

        if self.config.aggregation_message_type == 'sum':
            self.aggregation_message_fn = jraph.segment_sum
        elif self.config.aggregation_message_type == 'mean':
            self.aggregation_message_fn = jraph.segment_mean
        else:
            raise ValueError(
                f'Aggregation type {self.config.aggregation_message_type} '
                f'not recognized')

        if self.config.aggregation_readout_type == 'sum':
            self.aggregation_readout_fn = jraph.segment_sum
        elif self.config.aggregation_readout_type == 'mean':
            self.aggregation_readout_fn = jraph.segment_mean
        else:
            raise ValueError(
                f'Aggregation type {self.config.aggregation_readout_type} '
                f'not recognized')

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
            net = jraph.GraphNetwork(
                update_node_fn=get_node_update_fn(
                    self.config.latent_size, self.config.hk_init),
                update_edge_fn=get_edge_update_fn(
                    self.config.latent_size, self.config.hk_init),
                update_global_fn=None,
                aggregate_edges_for_nodes_fn=self.aggregation_message_fn)
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
            aggregate_nodes_for_globals_fn=self.aggregation_readout_fn)
        # Apply readout function on graph.
        graphs = net_readout(graphs)
        # Right now we are getting global tasks. If we wanted to get node
        # level tasks we could access the node features of the graphs here.
        return graphs

"""Graph Convolutional Neural Network with Kipf and Welling convolutions.
We modify and simplify the convolutions to work with directed (non-symmetric)
adjacency matrices."""

from jraph_MPEU.models import MLP, shifted_softplus, get_embedder
from jraph_MPEU.models import get_readout_node_update_fn
from jraph_MPEU.models import get_readout_global_fn


def get_node_update_fn(
        latent_size, hk_init, use_layer_norm, activation, dropout_rate,
        mlp_depth):
    """Return the node update function.

    Wrapper function so that we can interface with jraph.

    Args:
        latent_size: The size of the node feature vector and message vector.
        hk_init: The weight intializer function for the node feature vector
            update network.
        use_layer_norm: whether layer normalization should be used.
        activation: activation function for MLPs.
        dropout_rate: dropout rate after every activation layer in MLP.
        mlp_depth: number of weight layers in each MLP.
    """
    def node_update_fn(nodes) -> jnp.ndarray:
        """Node update function for GCN. Inputs aggregated edge features
        (received attributes, which, without an edge update function, are just
        neighboring node features) into a customizable MLP function.
        """

        net = MLP(
            'node_update', [latent_size]*mlp_depth,
            use_layer_norm=use_layer_norm, dropout_rate=dropout_rate,
            activation=activation, activate_final=False, w_init=hk_init)
        
        node_update = net(nodes)
        return node_update
    return node_update_fn


class GCN_kipf:
    """Graph neural network class where we define the interactions/updates."""
    def __init__(self, config: ml_collections.ConfigDict, is_training=True):
        """Initialize the GNN using a config.

        Args:
            config: Configuration specifying GNN properties.
        """
        self.config = config
        self.is_training = is_training

        self.config.aggregation_message_type == 'sum'

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
            net = jraph.GraphConvolution(
                update_node_fn=get_node_update_fn(
                    self.config.latent_size, self.config.hk_init, self.norm,
                    self.activation, dropout_rate, self.config.mlp_depth),
                aggregate_nodes_fn=self.aggregation_message_fn,
                add_self_edges=True,
                symmetric_normalization=True)
            # Update the graphs by applying our message passing step on graphs.
            graphs = net(graphs)
        # Then after we're done with message passing steps, we intiialize
        # our readout function. aggregate_nodes_for_globals_fn is used to
        # aggregate features to pass to the update_global_fn.

        # get the right node output size depending if there is an extra MLP
        # after the node to global aggregation
        node_output_size = self.config.latent_size if self.config.extra_mlp else 1
        dropout_rate = self.config.dropout_rate if self.is_training else 0.0
        net_readout = jraph.GraphNetwork(
            update_node_fn=get_readout_node_update_fn(
                self.config.latent_size,
                self.config.hk_init,
                node_output_size),
            update_edge_fn=None,
            update_global_fn=get_readout_global_fn(
                latent_size=self.config.latent_size,
                dropout_rate=dropout_rate,
                extra_mlp=self.config.extra_mlp,
                label_type=self.config.label_type,
                is_training=self.is_training),
            aggregate_nodes_for_globals_fn=self.aggregation_readout_fn)
        # Apply readout function on graph.
        graphs = net_readout(graphs)
        # Right now we are getting global tasks. If we wanted to get node
        # level tasks we could access the node features of the graphs here.
        return graphs

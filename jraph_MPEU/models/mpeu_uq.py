"""Here we define an MPEU (see models/mpeu.py) with uncertainty quantification,
by making it output a dict as globals with predicted mean (mu) and variance (sigma).
"""

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import ml_collections

#from jraph_MPEU.utils import load_config
from jraph_MPEU.models.mlp import (
    MLP, activation_dict, aggregation_dict)
from jraph_MPEU.models.mpeu import (
    _get_embedder,
    _get_node_update_fn,
    _get_edge_update_fn
)


def _get_readout_node_update_fn(
        latent_size, hk_init, use_batch_norm,
        dropout_rate, activation, is_training):
    """Return readout node update function.

    This is a wrapper function for the readout_node_update_fn.

    Args:
        latent_size: The node feature vector dimensionality.
        hk_init: The weight initializer for the the readout NN.
        use_layer_norm: whether layer normalization should be used.
        dropout_rate: dropout rate after every activation layer in MLP.
        output_size: dimensionality of the final node embeddings. If 1, it is
            assumed that this is the final layer and in readout global function
            only the nodes are aggregated.
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
        net_mu = MLP(
            'readout_node',
            [latent_size//2, 1],
            use_layer_norm=False,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation=activation,
            activate_final=False,
            w_init=hk_init,
            is_training=is_training)
        net_sigma = MLP(
            'readout_node',
            [latent_size//2, 1],
            use_layer_norm=False,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation=activation,
            activate_final=False,
            w_init=hk_init,
            is_training=is_training)
        # Apply the networks to the nodes.
        mu = net_mu(nodes)
        sigma = net_sigma(nodes)
        # constrain predicted variance to be positive, non-zero:
        sigma = jax.nn.softplus(sigma) + 1e-6
        return {'mu': mu, 'sigma': sigma}
    return readout_node_update_fn


def _get_readout_global_fn():
    """Return the readout global function.

    Args:
        latent_size: size of hidden layers in MLPs.
        global_readout_mlp_layers: number of MLP layers applied after
            aggregating node features.
        dropout_rate: dropout rate after every activation layer in MLP.
        activation: activation function for MLPs.
        label_type: type of the label that is fitted. Determines output size.
    """
    def readout_global_fn(
            node_attributes, edge_attributes,
            global_attributes) -> jnp.ndarray:
        """Global readout function for graph net.

        If global_readout_mlp_layers=0, returns the aggregated node features.
        If global_readout_mlp_layers>0, pass the aggregated node features
        through mlp, and return the last activation.

        Args:
            node_attributes: Aggregated node feature vectors.
            edge_attributes: Not used but expected by jraph.
            global_attributes: Not used but expected by jraph.
            global_readout_mlp_layers: number of mlp layers applied after
                aggregating node features
        """
        # Delete unused arguments that are expected by jraph.
        del edge_attributes, global_attributes
        return node_attributes
    return readout_global_fn


class MPEU_uq:
    """Graph neural network class where we define the interactions/updates."""
    def __init__(self, config: ml_collections.ConfigDict, is_training=True):
        """Initialize the GNN using a config.

        Args:
            config: Configuration specifying GNN properties.
        """
        self.config = config
        self.use_layer_norm = config.use_layer_norm
        self.use_batch_norm = config.use_batch_norm
        self.is_training = is_training

        self.activation = activation_dict[self.config.activation_name]
        self.aggregation_message_fn = aggregation_dict[
            self.config.aggregation_message_type]
        self.aggregation_readout_fn = aggregation_dict[
            self.config.aggregation_readout_type]

    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Call function to do forward pass of the GNN.

        Args:
            graphs: The input graphs on which to do the forward pass.
            is_training: used by BatchNorm in MLPs

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
        dropout_rate = self.config.dropout_rate if self.is_training else 0.0

        graphs = graphs._replace(
            globals=jnp.zeros([graphs.n_node.shape[0], 1], dtype=np.float32))

        # convert vector edges e_ij=r_i-r_j (displacement vectors) to distances
        norm_ij = jnp.sqrt(jnp.sum(graphs.edges**2, axis=1))
        graphs = graphs._replace(edges=norm_ij)

        embedder = _get_embedder(
            self.config.latent_size, self.config.k_max, self.config.delta,
            self.config.mu_min, self.config.max_atomic_number,
            self.config.hk_init, self.use_layer_norm, self.use_batch_norm,
            self.activation, self.is_training
        )
        # Embed the graph with embedder functions (nodes and edges get
        # embedded).
        graphs = embedder(graphs)
        # Run a for loop over the number of message passing steps.
        for _ in range(self.config.message_passing_steps):
            # In each step, we apply first the edge updates, the message
            # updates and then the node updates and then global updates (which
            # we don't do yet).
            net = jraph.GraphNetwork(
                update_node_fn=_get_node_update_fn(
                    self.config.latent_size,
                    self.config.hk_init,
                    self.use_layer_norm,
                    self.use_batch_norm,
                    self.activation,
                    dropout_rate,
                    self.config.mlp_depth,
                    self.is_training),
                update_edge_fn=_get_edge_update_fn(
                    self.config.latent_size,
                    self.config.hk_init,
                    self.use_layer_norm,
                    self.use_batch_norm,
                    self.activation,
                    dropout_rate,
                    self.config.mlp_depth,
                    self.is_training),
                update_global_fn=None,
                aggregate_edges_for_nodes_fn=self.aggregation_message_fn)
            # Update the graphs by applying our message passing step on graphs.
            graphs = net(graphs)
        # Then after we're done with message passing steps, we intiialize
        # our readout function. Nodes are updated a final time with
        # readout_node_update_function. aggregate_nodes_for_globals_fn is used
        # to aggregate features to pass to the update_global_fn.

        # get the right node output size depending if there is an extra MLP
        # after the node to global aggregation
        net_readout = jraph.GraphNetwork(
            update_node_fn=_get_readout_node_update_fn(
                self.config.latent_size,
                self.config.hk_init,
                self.use_batch_norm,
                dropout_rate,
                self.activation,
                self.is_training),
            update_edge_fn=None,
            update_global_fn=_get_readout_global_fn(),
            aggregate_nodes_for_globals_fn=self.aggregation_readout_fn)
        # Apply readout function on graph.
        graphs = net_readout(graphs)
        # Right now we are getting global tasks. If we wanted to get node
        # level tasks we could access the node features of the graphs here.
        return graphs

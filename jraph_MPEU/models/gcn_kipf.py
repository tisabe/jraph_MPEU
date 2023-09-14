"""Graph Convolutional Neural Network with Kipf and Welling convolutions."""



def get_edge_update_fn(
        latent_size: int, hk_init, use_layer_norm, activation, dropout_rate,
        mlp_depth):
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
        use_layer_norm: whether layer normalization should be used.
        activation: activation function for MLPs.
        dropout_rate: dropout rate after every activation layer in MLP.
        mlp_depth: number of weight layers in each MLP.
    """
    def edge_update_fn(
            edge_message, sent_attributes, received_attributes,
            global_edge_attributes) -> jnp.ndarray:
        """Edge update and message function for graph net.

        This function takes in an edge (sender and receiver node). We want to
        only return the information that is sent out.

        Args:
            edge_message: Another word for edge attribute.
                dictionary containing two keys, the edge vector
                and the message corresponding to that edge. Both of size
                latent_size (size C in the paper). For Kipf and Welling this
                needs to contain the node vector for the reciever/sender to
                we can get the correct sent_atributes and received_attributes
                which need to be the aggregation of the node vectors in a
                neighbourhood.
            sent_attributes: To update the rieceiving node vector we need
                the aggregated node vectors of the neighbourhood of the
                receiving node. So sent attributes needs to be this aggregation.
            receive_attributes: We don't need this here since.
            global_edge_attributes: Not used but expected by jraph function.
        """
        del global_edge_attributes, receive_attributes  # Not used.


        return sent_attributes

        # # Then element wise multiply together the output of the
        # # net_emessage_edge and the output from feeding the sending/receiving
        # # node feature vectors to the net_message node.
        # edge_message['messages'] = jnp.multiply(
        #     edges_new, nodes_new)
        # return edge_message
    return edge_update_fn

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
    def node_update_fn(
            nodes, sent_attributes,
            received_attributes, global_attributes) -> jnp.ndarray:
        """Node update function for graph net.

        Takes the previous node feature vector and the sum of incoming
        messages. Two layer NN on the sum of incoming messages at t+1 and adds
        this to the previous node feature vector at t. This gives us new node
        feature vector for time t+1.

        See Equation (9) in the PB Jorgensen paper.

        We specify that the received attributes when the are aggregated they
        use a specific aggregation function. We set this as the mean in the
        GCN class.

        Args:
            nodes: Node feature vector at previous iteration (h_i(t-1)).
            sent_attributes: Not used, should be the smae as recieved
                attributes.
            received_attributes: This is the aggregation of incoming edge
                attributes. recived_attributes need to be edge attributes
                that contain the node vectors of the neighbourhood + self
                interaction (not between periodic boundaries but inside the same
                unit cell). We need to change the edge update function to have
                this work.

        """
        # For testing purpopses we should check this.
        # assert sent_attributes == received_attributes

        # These input arguments are not used but expected by jraph.
        del sent_attributes, global_attributes
        # First layer is FC with shifted_softplus activation. Second layer
        # is linear.
        net = MLP(
            'node_update', [latent_size]*mlp_depth,
            use_layer_norm=use_layer_norm, dropout_rate=dropout_rate,
            activation=activation, activate_final=False, w_init=hk_init)
        # Get the messages term of Equation 9 which is the net applied to
        # the aggregation of incoming messages to the node.
        node_update = net(received_attributes['messages'])
        # Add the previous value of the node feature vector.
        return nodes + node_update
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

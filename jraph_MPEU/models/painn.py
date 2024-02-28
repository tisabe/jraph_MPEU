"""Define the PaiNN layer as defined in

Schütt, Kristof, Oliver Unke, and Michael Gastegger. "Equivariant message
passing for the prediction of tensorial properties and molecular spectra."
International Conference on Machine Learning. PMLR, 2021.

The implementation in the JAX framework is taken from
https://github.com/gerkone/painn-jax/tree/main according to the licence below.

Some modifications are made in order to integrate the functional behavior of
this model with the other models.
"""

# MIT License

# Copyright (c) 2023 Gianluca Galletti

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Callable, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph


class NodeFeatures(NamedTuple):
    """Simple container for scalar and vectorial node features."""

    s: jnp.ndarray = None
    v: jnp.ndarray = None


ReadoutFn = Callable[[jraph.GraphsTuple], Tuple[jnp.ndarray, jnp.ndarray]]
ReadoutBuilderFn = Callable[..., ReadoutFn]


def cosine_cutoff(cutoff: float) -> Callable[[jnp.ndarray], Callable]:
    r"""Behler-style cosine cutoff.

    .. math::
        f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
            & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.
    """
    hk.set_state("cutoff", cutoff)
    cutoff = hk.get_state("cutoff")

    def _cutoff(x: jnp.ndarray) -> jnp.ndarray:
        # Compute values of cutoff function
        cuts = 0.5 * (jnp.cos(x * jnp.pi / cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        mask = jnp.array(x < cutoff, dtype=jnp.float32)
        return cuts * mask

    return _cutoff


def gaussian_rbf(
    n_rbf: int,
    cutoff: float,
    start: float = 0.0,
    centered: bool = False,
    learnable: bool = False,
) -> Callable[[jnp.ndarray], Callable]:
    r"""Gaussian radial basis functions.

    Args:
        n_rbf: total number of Gaussian functions, :math:`N_g`.
        cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
        start: center of first Gaussian function, :math:`\mu_0`.
        learnable: If True, widths and offset of Gaussian functions learnable.
    """
    if centered:
        widths = jnp.linspace(start, cutoff, n_rbf)
        offset = jnp.zeros_like(widths)
    else:
        offset = jnp.linspace(start, cutoff, n_rbf)
        width = jnp.abs(cutoff - start) / n_rbf * jnp.ones_like(offset)

    if learnable:
        widths = hk.get_parameter(
            "widths", width.shape, width.dtype, init=lambda *_: width
        )
        offsets = hk.get_parameter(
            "offset", offset.shape, offset.dtype, init=lambda *_: offset
        )
    else:
        hk.set_state("widths", jnp.array([width]))
        hk.set_state("offsets", jnp.array([offset]))
        widths = hk.get_state("widths")
        offsets = hk.get_state("offsets")

    def _rbf(x: jnp.ndarray) -> jnp.ndarray:
        coeff = -0.5 / jnp.power(widths, 2)
        diff = x[..., jnp.newaxis] - offsets
        return jnp.exp(coeff * jnp.power(diff, 2))

    return _rbf


def bessel_rbf(n_rbf: int, cutoff: float) -> Callable[[jnp.ndarray], Callable]:
    r"""Sine for radial basis functions with coulomb decay (0th order bessel).

    Args:
        n_rbf: total number of Bessel functions, :math:`N_g`.
        cutoff: center of last Bessel function, :math:`\mu_{N_g}`

    References:
        [#dimenet] Klicpera, Groß, Günnemann:
        Directional message passing for molecular graphs.
        ICLR 2020
    """
    # compute offset and width of Gaussian functions
    freqs = jnp.arange(1, n_rbf + 1) * jnp.pi / cutoff
    hk.set_state("freqs", freqs)
    freqs = hk.get_state("freqs")

    def _rbf(self, x: jnp.ndarray) -> jnp.ndarray:
        ax = x[..., None] * self.freqs
        sinax = jnp.sin(ax)
        norm = jnp.where(x == 0, 1.0, x)
        y = sinax / norm[..., None]
        return y

    return _rbf


class LinearXav(hk.Linear):
    """Linear layer with Xavier init. Avoid distracting 'w_init' everywhere."""

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        super().__init__(output_size, with_bias, w_init, b_init, name)


class GatedEquivariantBlock(hk.Module):
    """Gated equivariant block (restricted to vectorial features). FIG 3 in the paper.

    References:
        [#painn1] Schütt, Unke, Gastegger:
        Equivariant message passing for the prediction of tensorial properties and
        molecular spectra.
        ICML 2021
    """

    def __init__(
        self,
        hidden_size: int,
        scalar_out_channels: int,
        vector_out_channels: int,
        activation: Callable = jax.nn.silu,
        scalar_activation: Callable = None,
        eps: float = 1e-8,
        name: str = "gated_equivariant_block",
    ):
        super().__init__(name)

        assert scalar_out_channels > 0 and vector_out_channels > 0
        self._scalar_out_channels = scalar_out_channels
        self._vector_out_channels = vector_out_channels
        self._eps = eps

        self.vector_mix_net = LinearXav(
            2 * vector_out_channels,
            with_bias=False,
            name="vector_mix_net",
        )
        self.gate_block = hk.Sequential(
            [
                LinearXav(hidden_size),
                activation,
                LinearXav(scalar_out_channels + vector_out_channels),
            ],
            name="scalar_gate_net",
        )
        self.scalar_activation = scalar_activation

    def __call__(
        self, s: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        v_l, v_r = jnp.split(self.vector_mix_net(v), 2, axis=-1)

        v_r_norm = jnp.sqrt(jnp.sum(v_r**2, axis=-2) + self._eps)
        gating_scalars = jnp.concatenate([s, v_r_norm], axis=-1)
        s, _, v_gate = jnp.split(
            self.gate_block(gating_scalars),
            [self._scalar_out_channels, self._vector_out_channels],
            axis=-1,
        )
        # scale the vectors by the gating scalars
        v = v_l * v_gate[:, jnp.newaxis]

        if self.scalar_activation:
            s = self.scalar_activation(s)

        return s, v


def pooling(
    graph: jraph.GraphsTuple,
    aggregate_fn: Callable = jraph.segment_sum,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pools over graph nodes with the specified aggregation.

    Args:
        graph: Input graph
        aggregate_fn: function used to update pool over the nodes

    Returns:
        The pooled graph nodes.
    """
    n_graphs = graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graphs)
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]
    batch = jnp.repeat(graph_idx, graph.n_node, axis=0, total_repeat_length=sum_n_node)

    s, v = None, None
    if graph.nodes.s is not None:
        s = aggregate_fn(graph.nodes.s, batch, n_graphs)
    if graph.nodes.v is not None:
        v = aggregate_fn(graph.nodes.v, batch, n_graphs)

    return s, v


def PaiNNReadout(
    hidden_size: int,
    task: str,
    pool: str,
    out_channels: int = 1,
    activation: Callable = jax.nn.silu,
    blocks: int = 2,
    eps: float = 1e-8,
) -> ReadoutFn:
    """
    PaiNN readout block.

    Args:
        hidden_size: Number of hidden channels.
        task: Task to perform. Either "node" or "graph".
        pool: pool method. Either "sum" or "avg".
        scalar_out_channels: Number of scalar/vector output channels.
        activation: Activation function.
        blocks: Number of readout blocks.
    """

    assert task in ["node", "graph"], "task must be node or graph"
    assert pool in ["sum", "avg"], "pool must be sum or avg"
    if pool == "avg":
        pool_fn = jraph.segment_mean
    if pool == "sum":
        pool_fn = jraph.segment_sum

    def _readout(graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        s, v = graph.nodes
        s = jnp.squeeze(s)
        for i in range(blocks - 1):
            ith_hidden_size = hidden_size // 2 ** (i + 1)
            s, v = GatedEquivariantBlock(
                hidden_size=ith_hidden_size * 2,
                scalar_out_channels=ith_hidden_size,
                vector_out_channels=ith_hidden_size,
                activation=activation,
                eps=eps,
                name=f"readout_block_{i}",
            )(s, v)

        if task == "graph":
            graph = graph._replace(nodes=NodeFeatures(s, v))
            s, v = pooling(graph, aggregate_fn=pool_fn)

        s, v = GatedEquivariantBlock(
            hidden_size=ith_hidden_size,
            scalar_out_channels=out_channels,
            vector_out_channels=out_channels,
            activation=activation,
            eps=eps,
            name="readout_block_out",
        )(s, v)

        return jnp.squeeze(s), jnp.squeeze(v)

    return _readout


class PaiNNLayer(hk.Module):
    """PaiNN interaction block."""

    def __init__(
        self,
        hidden_size: int,
        layer_num: int,
        activation: Callable = jax.nn.silu,
        blocks: int = 2,
        aggregate_fn: Callable = jraph.segment_sum,
        eps: float = 1e-8,
    ):
        """
        Initialize the PaiNN layer, made up of an interaction block and a mixing block.

        Args:
            hidden_size: Number of node features.
            activation: Activation function.
            layer_num: Numbering of the layer.
            blocks: Number of layers in the context networks.
            aggregate_fn: Function to aggregate the neighbors.
            eps: Constant added in norm to prevent derivation instabilities.
        """
        super().__init__(f"layer_{layer_num}")
        self._hidden_size = hidden_size
        self._eps = eps
        self._aggregate_fn = aggregate_fn

        # inter-particle context net
        self.interaction_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (blocks - 1)
            + [LinearXav(3 * hidden_size)],
            name="interaction_block",
        )

        # intra-particle context net
        self.mixing_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (blocks - 1)
            + [LinearXav(3 * hidden_size)],
            name="mixing_block",
        )

        # vector channel mix
        self.vector_mixing_block = LinearXav(
            2 * hidden_size,
            with_bias=False,
            name="vector_mixing_block",
        )

    def _message(
        self,
        s: jnp.ndarray,
        v: jnp.ndarray,
        dir_ij: jnp.ndarray,
        Wij: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Message/interaction. Inter-particle.

        Args:
            s (jnp.ndarray): Input scalar features (n_nodes, 1, hidden_size).
            v (jnp.ndarray): Input vector features (n_nodes, 3, hidden_size).
            dir_ij (jnp.ndarray): Direction of the edge (n_edges, 3).
            Wij (jnp.ndarray): Filter (n_edges, 1, 3 * hidden_size).
            senders (jnp.ndarray): Index of the sender node.
            receivers (jnp.ndarray): Index of the receiver node.

        Returns:
            Aggregated messages after interaction.
        """
        x = self.interaction_block(s)

        xj = x[receivers]
        vj = v[receivers]

        ds, dv1, dv2 = jnp.split(Wij * xj, 3, axis=-1)  # (n_edges, 1, hidden_size)
        n_nodes = tree.tree_leaves(s)[0].shape[0]
        dv = dv1 * dir_ij[..., jnp.newaxis] + dv2 * vj  # (n_edges, 3, hidden_size)
        # aggregate scalars and vectors
        ds = self._aggregate_fn(ds, senders, n_nodes)
        dv = self._aggregate_fn(dv, senders, n_nodes)

        s = s + jnp.clip(ds, -1e2, 1e2)
        v = v + jnp.clip(dv, -1e2, 1e2)

        return s, v

    def _update(
        self, s: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update/mixing. Intra-particle.

        Args:
            s (jnp.ndarray): Input scalar features (n_nodes, 1, hidden_size).
            v (jnp.ndarray): Input vector features (n_nodes, 3, hidden_size).

        Returns:
            Node features after update.
        """
        v_l, v_r = jnp.split(self.vector_mixing_block(v), 2, axis=-1)
        v_norm = jnp.sqrt(jnp.sum(v_r**2, axis=-2, keepdims=True) + self._eps)

        ts = jnp.concatenate([s, v_norm], axis=-1)  # (n_nodes, 1, 2 * hidden_size)
        ds, dv, dsv = jnp.split(self.mixing_block(ts), 3, axis=-1)
        dv = v_l * dv
        dsv = dsv * jnp.sum(v_r * v_l, axis=1, keepdims=True)

        s = s + jnp.clip(ds + dsv, -1e2, 1e2)
        v = v + jnp.clip(dv, -1e2, 1e2)
        return s, v

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        Wij: jnp.ndarray,
    ):
        """Compute interaction output.

        Args:
            graph (jraph.GraphsTuple): Input graph.
            Wij (jnp.ndarray): Filter.

        Returns:
            atom features after interaction
        """
        s, v = graph.nodes
        s, v = self._message(s, v, graph.edges, Wij, graph.senders, graph.receivers)
        s, v = self._update(s, v)
        return graph._replace(nodes=NodeFeatures(s=s, v=v))


def get_painn(config):
    """Return the painn model object with parameters from config."""
    return PaiNN(
        hidden_size=config.latent_size,
        n_layers=config.message_passing_steps,
        radial_basis_fn=gaussian_rbf,
        cutoff_fn=None,
        radius=config.cutoff_radius,
        task="graph",
        pool=config.aggregation_readout_type,
        out_channels=1,
        readout_fn=PaiNNReadout
    )


class PaiNN(hk.Module):
    """PaiNN - polarizable interaction neural network.

    References:
        [#painn1] Schütt, Unke, Gastegger:
        Equivariant message passing for the prediction of tensorial properties and
        molecular spectra.
        ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        radial_basis_fn: Callable,
        *args,
        cutoff_fn: Optional[Callable] = None,
        radius: float = 5.0,
        n_rbf: int = 20,
        activation: Callable = jax.nn.silu,
        node_type: str = "discrete",
        task: str = "node",
        pool: str = "sum",
        out_channels: Optional[int] = None,
        readout_fn: ReadoutBuilderFn = PaiNNReadout,
        max_z: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        """Initialize the model.

        Args:
            hidden_size: Determines the size of each embedding vector.
            n_layers: Number of interaction blocks.
            radial_basis_fn: Expands inter-particle distances in a basis set.
            cutoff_fn: Cutoff method. None means no cutoff.
            radius: Cutoff radius.
            n_rbf: Number of radial basis functions.
            activation: Activation function.
            node_type: Type of node features. Either "discrete" or "continuous".
            task: Regression task to perform. Either "node"-wise or "graph"-wise.
            pool: Node readout pool method. Only used in "graph" tasks.
            out_channels: Number of output scalar/vector channels. Used in readout.
            readout_fn: Readout function. If None, use default PaiNNReadout.
            max_z: Maximum atomic number. Used in discrete node feature embedding.
            shared_interactions: If True, share the weights across interaction blocks.
            shared_filters: If True, share the weights across filter networks.
            eps: Constant added in norm to prevent derivation instabilities.
        """
        super().__init__("painn")

        assert node_type in [
            "discrete",
            "continuous",
        ], "node_type must be discrete or continuous"
        assert task in ["node", "graph"], "task must be node or graph"
        assert radial_basis_fn is not None, "A radial_basis_fn must be provided"

        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._eps = eps
        self._node_type = node_type
        self._shared_filters = shared_filters
        self._shared_interactions = shared_interactions

        self.cutoff_fn = cutoff_fn(radius) if cutoff_fn else None
        self.radial_basis_fn = radial_basis_fn(n_rbf, radius)

        if node_type == "discrete":
            self.scalar_emb = hk.Embed(
                max_z,
                hidden_size,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                name="scalar_embedding",
            )
        else:
            self.scalar_emb = LinearXav(hidden_size, name="scalar_embedding")
        # mix vector channels (only used if vector features are present in input)
        self.vector_emb = LinearXav(
            hidden_size, with_bias=False, name="vector_embedding"
        )

        if shared_filters:
            self.filter_net = LinearXav(3 * hidden_size, name="filter_net")
        else:
            self.filter_net = LinearXav(n_layers * 3 * hidden_size, name="filter_net")

        if self._shared_interactions:
            self.layers = [PaiNNLayer(hidden_size, 0, activation, eps=eps)] * n_layers
        else:
            self.layers = [
                PaiNNLayer(hidden_size, i, activation, eps=eps) for i in range(n_layers)
            ]

        self.readout = None
        if out_channels is not None and readout_fn is not None:
            self.readout = readout_fn(
                *args,
                hidden_size,
                task,
                pool,
                out_channels=out_channels,
                activation=activation,
                eps=eps,
                **kwargs,
            )

    def _embed(self, graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Embed the input nodes."""
        # extension to work with graphs generated from input pipeline
        graph = graph._replace(
            nodes=NodeFeatures(graph.nodes, None))
        # embeds scalar features
        s = graph.nodes.s
        if self._node_type == "continuous":
            # e.g. velocities
            s = jnp.asarray(s, dtype=jnp.float32)
            if len(s.shape) == 1:
                s = s[:, jnp.newaxis]
        if self._node_type == "discrete":
            # e.g. atomic numbers
            s = jnp.asarray(s, dtype=jnp.int32)
        s = self.scalar_emb(s)[:, jnp.newaxis]  # (n_nodes, 1, hidden_size)

        # embeds vector features
        if graph.nodes.v is not None:
            # initialize the vector with the global positions
            v = graph.nodes.v
            v = self.vector_emb(v)  # (n_nodes, 3, hidden_size)
        else:
            # if no directional info, initialize the vector with zeros (as in the paper)
            v = jnp.zeros((s.shape[0], 3, s.shape[-1]))

        return graph._replace(nodes=NodeFeatures(s=s, v=v))

    def _get_filters(self, norm_ij: jnp.ndarray) -> jnp.ndarray:
        r"""Compute the rotationally invariant filters :math:`W_s`.

        .. math::
            W_s = MLP(RBF(\|\vector{r}_{ij}\|)) * f_{cut}(\|\vector{r}_{ij}\|)
        """
        phi_ij = self.radial_basis_fn(norm_ij)
        if self.cutoff_fn is not None:
            norm_ij = self.cutoff_fn(norm_ij)
        # compute filters
        filters = (
            self.filter_net(phi_ij) * norm_ij[:, jnp.newaxis]
        )  # (n_edges, 1, n_layers * 3 * hidden_size)
        # split into layer-wise filters
        if self._shared_filters:
            filter_list = [filters] * self._n_layers
        else:
            filter_list = jnp.split(filters, self._n_layers, axis=-1)
        return filter_list

    def __call__(self, graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute representations/embeddings.

        Args:
            inputs: GraphsTuple. The nodes should cointain a NodeFeatures object with
                - scalar feature of the shape (n_atoms, n_features)
                - vector feature of the shape (n_atoms, 3, n_features)

        Returns:
            Tuple with scalar and vector representations/embeddings.
        """
        # compute atom and pair features
        norm_ij = jnp.sqrt(jnp.sum(graph.edges**2, axis=1, keepdims=True) + self._eps)
        # edge directions
        # NOTE: assumes edge features are displacement vectors.
        dir_ij = graph.edges / (norm_ij + self._eps)
        graph = graph._replace(edges=dir_ij)

        # compute filters (r_ij track in message block from the paper)
        filter_list = self._get_filters(norm_ij)  # list (n_edges, 1, 3 * hidden_size)

        # embeds node scalar features (and vector, if present)
        graph = self._embed(graph)

        # message passing
        for n, layer in enumerate(self.layers):
            graph = layer(graph, filter_list[n])

        if self.readout is not None:
            # return decoded representations
            s, v = self.readout(graph)
        else:
            # return representations (last layer embedding)
            s, v = jnp.squeeze(graph.nodes.s), jnp.squeeze(graph.nodes.v)
        return s, v

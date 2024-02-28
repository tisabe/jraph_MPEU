"""Test defining and output of the PaiNN model."""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import ml_collections

from jraph_MPEU.models.painn import (
    PaiNN, get_painn, gaussian_rbf, cosine_cutoff, PaiNNReadout, NodeFeatures)


def rotation_matrix_x(alpha):
    """Return a rotation matrix with a rotation of alph radians on the x-axis."""
    mat = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -1*np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )
    return mat


class UnitTest(unittest.TestCase):
    """Unit test functions in painn.py."""
    def setUp(self):
        self.config = ml_collections.ConfigDict()
        self.config.latent_size=64
        self.config.message_passing_steps=3
        self.config.cutoff_radius=6.0
        self.config.aggregation_readout_type="sum"

    def test_equivariance(self):
        """Test that the model is equivariant by rotating the edges."""
        n_node = 4
        # dummy graph as input
        graph = jraph.GraphsTuple(
            nodes=jnp.asarray([1, 2]*int(n_node/2)),
            edges=jnp.ones((n_node, 3)),
            senders=jnp.arange(0, n_node),
            receivers=jnp.arange(0, n_node),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_node]),
            globals=None
        )
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        def painn(x):
            return PaiNN(
                hidden_size=8,
                n_layers=3,
                max_z=20,
                node_type="discrete",
                radial_basis_fn=gaussian_rbf,
                cutoff_fn=cosine_cutoff,
                task="graph",
                pool="sum",
                radius=5.,
                n_rbf=20,
                out_channels=1,
                readout_fn=PaiNNReadout,
                #readout_fn=None
            )(x)

        net = hk.without_apply_rng(hk.transform_with_state(painn))
        params, state = net.init(init_rng, graph) # create weights etc. for the model

        graph_pred, state = net.apply(params, state, graph)

        s, v = graph_pred

        rot_mat = rotation_matrix_x(1)
        v_rot = rot_mat @ np.transpose(v)

        edges_rot = rot_mat @ np.transpose(graph.edges)
        edges_rot = np.transpose(edges_rot)
        graph = graph._replace(edges=edges_rot)
        graph_pred, state = net.apply(params, state, graph)
        rot_s, rot_v = graph_pred

        np.testing.assert_allclose(rot_v, v_rot, rtol=1e-6)
        np.testing.assert_allclose(s, rot_s, rtol=1e-6)




if __name__ == '__main__':
    unittest.main()

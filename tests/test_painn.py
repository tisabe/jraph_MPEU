"""Test defining and output of the PaiNN model."""

import unittest
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import ml_collections

from jraph_MPEU import train
from jraph_MPEU.models.painn import (
    PaiNN, get_painn, gaussian_rbf, cosine_cutoff, PaiNNReadout, NodeFeatures,
    StandardReadout)
from jraph_MPEU_configs import test_painn as cfg_painn


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

    def test_equivariance_graph_readout(self):
        """Test that the graph readout is equivariant upon rotating the edges."""
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
                readout_fn=PaiNNReadout
            )(x)

        net = hk.without_apply_rng(hk.transform_with_state(painn))
        params, state = net.init(init_rng, graph) # create weights etc. for the model

        graph_pred, state = net.apply(params, state, graph)

        s, v = graph_pred.globals.s, graph_pred.globals.v

        rot_mat = rotation_matrix_x(1)
        v_rot = rot_mat @ np.transpose(v)

        edges_rot = rot_mat @ np.transpose(graph.edges)
        edges_rot = np.transpose(edges_rot)
        graph = graph._replace(edges=edges_rot)
        graph_pred, state = net.apply(params, state, graph)
        rot_s, rot_v = graph_pred.globals.s, graph_pred.globals.v

        np.testing.assert_allclose(rot_v, v_rot, rtol=1e-6)
        np.testing.assert_allclose(s, rot_s, rtol=1e-6)

    def test_equivariance_node_readout(self):
        """Test that the graph readout is equivariant upon rotating the edges."""
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
                task="node",
                pool="sum",
                radius=5.,
                n_rbf=20,
                out_channels=1,
                readout_fn=PaiNNReadout
            )(x)

        net = hk.without_apply_rng(hk.transform_with_state(painn))
        params, state = net.init(init_rng, graph) # create weights etc. for the model

        graph_pred, state = net.apply(params, state, graph)

        s, v = graph_pred.nodes.s, graph_pred.nodes.v

        rot_mat = rotation_matrix_x(1)
        v_rot = rot_mat @ np.transpose(v)
        v_rot = np.transpose(v_rot)

        edges_rot = rot_mat @ np.transpose(graph.edges)
        edges_rot = np.transpose(edges_rot)
        graph = graph._replace(edges=edges_rot)
        graph_pred, state = net.apply(params, state, graph)
        rot_s, rot_v = graph_pred.nodes.s, graph_pred.nodes.v

        np.testing.assert_allclose(rot_v, v_rot, rtol=1e-6)
        np.testing.assert_allclose(s, rot_s, rtol=1e-6)

    def test_train_painn(self):
        """Test training the painn model."""
        config = cfg_painn.get_config()
        config.model_str = 'PaiNN'
        config.cutoff_radius = 5.
        with tempfile.TemporaryDirectory() as test_dir:
            evaluater, lowest_val_loss = train.train_and_evaluate(
                config, test_dir)

            self.assertIsInstance(evaluater.best_state, dict)
            print(evaluater.loss_dict['train'])

    def test_mean_aggregation(self):
        """Test that the readout value is invariant with number of atoms,
        when pooling method is 'mean'."""
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
                pool="mean",
                radius=5.,
                n_rbf=20,
                out_channels=1,
                readout_fn=StandardReadout
            )(x)

        net = hk.without_apply_rng(hk.transform_with_state(painn))
        params, state = net.init(init_rng, graph) # create weights etc. for the model

        graph_pred, state = net.apply(params, state, graph)
        pred = graph_pred.globals

        n_node = 2*n_node
        graph_double = jraph.GraphsTuple(
            nodes=jnp.asarray([1, 2]*int(n_node/2)),
            edges=jnp.ones((n_node, 3)),
            senders=jnp.arange(0, n_node),
            receivers=jnp.arange(0, n_node),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_node]),
            globals=None
        )
        graph_double_pred, state = net.apply(params, state, graph_double)
        pred_double = graph_double_pred.globals
        self.assertAlmostEqual(pred, pred_double, places=6)

    def test_sum_aggregation(self):
        """Test that the readout value scales with number of atoms,
        when pooling method is 'sum'."""
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
                readout_fn=StandardReadout
            )(x)

        net = hk.without_apply_rng(hk.transform_with_state(painn))
        params, state = net.init(init_rng, graph) # create weights etc. for the model

        graph_pred, state = net.apply(params, state, graph)
        pred = graph_pred.globals

        n_node = 2*n_node
        graph_double = jraph.GraphsTuple(
            nodes=jnp.asarray([1, 2]*int(n_node/2)),
            edges=jnp.ones((n_node, 3)),
            senders=jnp.arange(0, n_node),
            receivers=jnp.arange(0, n_node),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_node]),
            globals=None
        )
        graph_double_pred, state = net.apply(params, state, graph_double)
        pred_double = graph_double_pred.globals

        self.assertAlmostEqual(2*pred, pred_double, places=6)


if __name__ == '__main__':
    unittest.main()

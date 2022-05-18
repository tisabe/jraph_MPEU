"""Test functions in the asedb_to_graphs module."""

import unittest

import numpy as np
import jax.numpy as jnp
from ase import Atoms

from asedb_to_graphs import (
    get_graph_fc,
    get_graph_cutoff,
    get_graph_knearest
)

def dist_matrix(position_matrix):
    '''Return the pairwise distance matrix of positions in euclidian space.

    See this link:
    https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow

    We multiply the position matrices together. And then reduce the sum by summing
    over the distance to the origin origin axis of the squared position matrix.

    Args:
      position_matrix: Position matrix given as an array.
        Size: (number of positions, dimension of positions)

    Returns:
      Euclidian distances between nodes as a jnp array.
        Size: (number of positions, number of positions)
    '''

    row_norm_squared = jnp.sum(position_matrix * position_matrix, axis=-1)
    # Turn r into column vector
    row_norm_squared = jnp.reshape(row_norm_squared, [-1, 1])
    # 2*pos*potT
    distance_matrix = 2 * jnp.matmul(position_matrix, jnp.transpose(position_matrix))
    distance_matrix = row_norm_squared + jnp.transpose(row_norm_squared) - distance_matrix
    distance_matrix = jnp.abs(distance_matrix)  # to avoid negative numbers before sqrt
    return jnp.sqrt(distance_matrix)


class TestGraphsFunctions(unittest.TestCase):
    def test_graph_fc(self):
        atoms = Atoms('H5')
        num_nodes = 5
        dimensions = 3
        np.random.seed(42)
        position_matrix = np.random.randint(0, 10, size=(num_nodes, dimensions))
        distances = dist_matrix(position_matrix)
        atoms.set_positions(position_matrix)

        nodes, pos, edges, senders, receivers = get_graph_fc(atoms)

        expected_edges = []
        expected_senders = []
        expected_receivers = []

        for receiver in range(num_nodes):
            for sender in range(num_nodes):
                if (sender != receiver):
                    expected_edges.append(distances[sender, receiver])
                    expected_senders.append(sender)
                    expected_receivers.append(receiver)

        np.testing.assert_array_equal(np.array([1]*num_nodes), nodes)
        np.testing.assert_array_equal(pos, position_matrix)
        np.testing.assert_array_almost_equal(np.array(expected_edges), edges)
        np.testing.assert_array_equal(np.array(expected_senders), senders)
        np.testing.assert_array_equal(np.array(expected_receivers), receivers)


    def test_catch_fc_with_pbc(self):
        """Test that trying to make a fully connected graph from atoms with periodic 
        boundary conditions raises an exception."""
        atoms = Atoms('H5', pbc=True)
        num_nodes = 5
        dimensions = 3
        np.random.seed(42)
        position_matrix = np.random.randint(0, 10, size=(num_nodes, dimensions))
        atoms.set_positions(position_matrix)

        with self.assertRaises(Exception, msg='PBC not allowed for fully connected graph.'):
            get_graph_fc(atoms)


    def test_k_nn_random(self):
        atoms = Atoms('H5')
        num_nodes = 5
        dimensions = 3
        k = 3
        np.random.seed(42)
        position_matrix = np.random.randint(0, 10, size=(num_nodes, dimensions))
        distances = dist_matrix(position_matrix)
        atoms.set_positions(position_matrix)
        nodes, pos, edges, senders, receivers = get_graph_knearest(atoms, k)

        expected_edges = []
        expected_senders = []
        expected_receivers = []

        for row in range(num_nodes):
            idx_list = []
            last_idx = 0
            for ik in range(k):
                min_val_last = 9999.9  # temporary last saved minimum value, initialized to high value
                for col in range(num_nodes):
                    if col == row or (col in idx_list):
                        continue    # do nothing on the diagonal, or if column has already been included
                    else:
                        val = distances[row, col]
                        if val < min_val_last:
                            min_val_last = val
                            last_idx = col
                idx_list.append(last_idx)
                expected_senders.append(last_idx)
                expected_receivers.append(row)

        for s, r in zip(expected_senders, expected_receivers):
            expected_edges.append(distances[s, r])

        np.testing.assert_array_equal(np.array([1]*num_nodes), nodes)
        np.testing.assert_array_equal(pos, position_matrix)
        np.testing.assert_array_almost_equal(np.array(expected_edges), edges)
        np.testing.assert_array_equal(np.array(expected_senders), senders)
        np.testing.assert_array_equal(np.array(expected_receivers), receivers)

    def test_get_cutoff_adj_from_dist_random(self):
        atoms = Atoms('H4')
        num_nodes = 4
        dimensions = 3
        cutoff = 0.7
        position_matrix = np.random.rand(num_nodes, dimensions)
        distances = dist_matrix(position_matrix)

        atoms.set_positions(position_matrix)
        nodes, pos, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)

        expected_edges = []
        expected_senders = []
        expected_receivers = []

        for sender in range(num_nodes):
            for receiver in range(num_nodes):
                if not (sender == receiver):
                    if distances[sender, receiver] < cutoff:
                        expected_edges.append(distances[sender, receiver])
                        expected_senders.append(sender)
                        expected_receivers.append(receiver)

        # edges might be arranged differently
        edges = np.sort(edges)
        expected_edges = np.sort(expected_edges)

        np.testing.assert_array_equal(np.array([1]*num_nodes), nodes)
        np.testing.assert_array_equal(pos, position_matrix)
        np.testing.assert_array_almost_equal(np.array(expected_edges), edges)
        self.assertCountEqual(np.array(expected_senders), senders)
        self.assertCountEqual(np.array(expected_receivers), receivers)


if __name__ == '__main__':
    unittest.main()

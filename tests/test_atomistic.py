"""Unit tests for jraph_MPEU.atomistic functions."""

from ase import Atoms
import numpy as np

from jraph_MPEU.atomistic import (
    get_neighborhood,
    atoms_to_graph
)


def test_molecule():
    """Produce a fully connected graph."""
    atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1)])
    pos = atoms.get_positions()
    senders, receivers, unit_shifts = get_neighborhood(
        atoms, cutoff=None, self_edges=False)
    np.testing.assert_array_equal(senders, [0, 1])
    np.testing.assert_array_equal(receivers, [1, 0])
    np.testing.assert_array_equal(unit_shifts, np.zeros_like(pos))


def test_1D():
    atoms = Atoms('Au',
        positions=[[0, 0, 0]],
        cell=[2.9, 0., 0.],
        pbc=[1, 0, 0])
    senders, receivers, unit_shifts = get_neighborhood(
        atoms, cutoff=3, self_edges=False)
    np.testing.assert_array_equal(senders, [0, 0])
    np.testing.assert_array_equal(receivers, [0, 0])
    np.testing.assert_array_equal(
        unit_shifts, np.array([[-1, 0, 0], [1, 0, 0]]))


def test_3D():
    atoms = Atoms('Au',
        positions=[[0, 0, 0]],
        cell=[1, 1, 1],
        pbc=[1, 1, 1])
    senders, receivers, unit_shifts = get_neighborhood(
        atoms, cutoff=1.1, self_edges=False)


def test_atoms_to_graph():
    atoms = Atoms('Au',
        positions=[[0, 0, 0]],
        cell=[1, 1, 1],
        pbc=[1, 1, 1])
    graph = atoms_to_graph(atoms, cutoff=1.1, self_edges=False)
    np.testing.assert_array_equal(
        graph.edges['unit_shifts'],
        np.array([
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]))


def test_k_nn_random(self):
    """Test generating a k-nearest neighbor graph from random atomic
    positions."""
    num_nodes = 5
    atoms = Atoms(f'H{num_nodes}')
    dimensions = 3
    k = 3
    position_matrix = self.rng.integers(0, 10, size=(num_nodes, dimensions))
    distances = dist_matrix(position_matrix)
    atoms.set_positions(position_matrix)
    nodes, pos, edges, senders, receivers = get_graph_knearest(atoms, k)

    expected_senders = []
    expected_receivers = []

    for row in range(num_nodes):
        idx_list = []
        last_idx = 0
        for _ in range(k):
            # temporary last saved minimum value, initialized to high value
            min_val_last = 9999.9
            for col in range(num_nodes):
                if col == row or (col in idx_list):
                    # do nothing on the diagonal,
                    # or if column has already been included
                    continue
                else:
                    val = distances[row, col]
                    if val < min_val_last:
                        min_val_last = val
                        last_idx = col
            idx_list.append(last_idx)
            expected_senders.append(last_idx)
            expected_receivers.append(row)

    expected_edges = position_matrix[expected_receivers] - position_matrix[expected_senders]
    # we only check distances exactly, since senders have an arbitrary
    # ordering because of the way neighborlists are built in ase
    dists = np.sqrt(np.sum(edges**2, axis=1))
    dists_expected = np.sqrt(np.sum(expected_edges**2, axis=1))
    np.testing.assert_array_equal(np.array(dists_expected), dists)
    self.assertTupleEqual(np.shape(nodes), (num_nodes,))
    self.assertTupleEqual(np.shape(pos), (num_nodes, dimensions))
    self.assertTupleEqual(np.shape(edges), (num_nodes*k, dimensions))
    self.assertTupleEqual(np.shape(senders), (num_nodes*k,))
    self.assertTupleEqual(np.shape(receivers), (num_nodes*k,))


def test_k_nn_pbc(self):
    """Test generating a k-nearest neighbor graph from random atomic
    positions with periodic boundary conditions."""
    cell_l = 2
    num_nodes = 5
    atoms = Atoms(f'H{num_nodes}', cell=[cell_l]*3, pbc=[1, 1, 1])
    dimensions = 3
    k = 3
    position_matrix = self.rng.integers(0, 10, size=(num_nodes, dimensions))
    atoms.set_positions(position_matrix)
    nodes, pos, edges, senders, receivers = get_graph_knearest(atoms, k)
    self.assertTupleEqual(np.shape(nodes), (num_nodes,))
    self.assertTupleEqual(np.shape(pos), (num_nodes, dimensions))
    self.assertTupleEqual(np.shape(edges), (num_nodes*k, dimensions))
    self.assertTupleEqual(np.shape(senders), (num_nodes*k,))
    self.assertTupleEqual(np.shape(receivers), (num_nodes*k,))
    # check that coordinates of pos have been wrapped to inside the cell
    for coordinate in pos.flatten():
        self.assertLessEqual(coordinate, cell_l)


def test_k_nn_too_far(self):
    """Test generating a k-nearest neighbor graph, but an exception is
    raised because the atoms are too far apart."""
    atoms = Atoms('H2')
    dimensions = 3
    scale = 10
    position_matrix = [[0]*dimensions, [scale]*dimensions]
    k = 1
    atoms.set_positions(position_matrix)
    with self.assertRaises(RuntimeError):
        _ = get_graph_knearest(atoms, k, initial_radius=scale/20)


if __name__ == "__main__":
    test_molecule()
    test_1D()
    test_3D()
    test_atoms_to_graph()

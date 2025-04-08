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


if __name__ == "__main__":
    test_molecule()
    test_1D()
    test_3D()
    test_atoms_to_graph()

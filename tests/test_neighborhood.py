"""Unit tests for jraph_MPEU.neighborhood functions."""

from ase import Atoms
import numpy as np

from jraph_MPEU.neighborhood import get_neighborhood


def test_molecule():
    atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1)])
    pos = atoms.get_positions()
    senders, receivers, dist_vecs, unit_shifts = get_neighborhood(
        atoms, cutoff=None, self_edges=False)
    np.testing.assert_array_equal(senders, [0, 1])
    np.testing.assert_array_equal(receivers, [1, 0])
    np.testing.assert_array_equal(dist_vecs, 
        [pos[1]-pos[0], pos[0]-pos[1]])
    np.testing.assert_array_equal(unit_shifts, np.zeros_like(pos))


def test_1D():
    atoms = Atoms('Au',
        positions=[[0, 10. / 2, 10. / 2]],
        cell=[2.9, 0., 0.],
        pbc=[1, 0, 0])
    senders, receivers, dist_vecs, unit_shifts = get_neighborhood(
        atoms, cutoff=3, self_edges=False)
    print(senders, receivers, dist_vecs, unit_shifts)


if __name__ == "__main__":
    test_molecule()
    test_1D()

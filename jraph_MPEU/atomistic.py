"""Module with neighborhood functions for graph edge generation."""

from typing import Optional, Tuple

import ase
import numpy as np
from matscipy.neighbours import neighbour_list
import jraph


def get_neighborhood_fc(
    atoms: ase.Atoms,
    self_edges: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    senders = []
    receivers = []
    pos = atoms.get_positions(wrap=True)
    n_atoms = len(atoms)
    unit_shifts = np.zeros((n_atoms, 3))

    for s in range(n_atoms):
        for r in range(n_atoms):
            # get all edges except self edges
            if (s != r) or self_edges:
                s_pos = pos[s]
                r_pos = pos[r]

                senders.append(s)
                receivers.append(r)
    return np.array(senders), np.array(receivers), unit_shifts


def get_neighborhood(
    atoms: ase.Atoms,
    cutoff: Optional[float] = None,
    self_edges: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns graph adjacency through senders, receivers and unit_shifts."""
    if cutoff is None:
        return get_neighborhood_fc(atoms, self_edges)
    else:
        cutoff = float(cutoff)
    if np.any(atoms.cell == 0):
        #atoms.set_cell(atoms.cell + np.diag(np.diagonal(atoms.cell == 0)))
        atoms.set_cell(atoms.cell + np.diag(np.logical_not(atoms.pbc)))
    senders, receivers, unit_shifts = neighbour_list('ijS', atoms, cutoff)

    if not self_edges:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = senders == receivers
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        senders = senders[keep_edge]
        receivers = receivers[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    return senders, receivers, unit_shifts


def atoms_to_graph(
    atoms: ase.Atoms,
    cutoff: Optional[float] = None,
    self_edges: bool = False,
    input_tree_def: Optional[dict] = None
) -> jraph.GraphsTuple:
    senders, receivers, unit_shifts = get_neighborhood(
        atoms, cutoff, self_edges)
    nodes = {
        'atomic_numbers': atoms.get_atomic_numbers(),
        'positions': atoms.get_positions()
    }
    edges = {'unit_shifts': unit_shifts}
    globals_ = {'cell': atoms.get_cell()}

    graph = jraph.GraphsTuple(
        n_node=np.asarray([len(atoms)]),
        n_edge=np.asarray([len(senders)]),
        nodes=nodes,
        edges=edges,
        globals=globals_,
        senders=senders,
        receivers=receivers)

    return graph
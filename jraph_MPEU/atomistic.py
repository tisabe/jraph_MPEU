"""Module with neighborhood functions for graph edge generation."""

from typing import Optional, Tuple

import ase
from ase.neighborlist import NeighborList
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


def get_graph_knearest(
        atoms: ase.Atoms, num_neighbors, initial_radius=3.0):
    """Return the graph features, with knearest adjacency.
    Inspired by https://github.com/peterbjorgensen/msgnet/blob/master/src/msgnet/dataloader.py
    """

    atoms.wrap() # put atoms inside unit cell by wrapping their positions
    atom_numbers = atoms.get_atomic_numbers()
    unitcell = atoms.get_cell()

    # We want to calculate k nearest neighbors, so we start within a sphere
    # with radius R. In this sphere we are calculating the number of neighbors,
    # if there are not enough, i.e. the number of neighbors within the sphere
    # is smaller than k, R is increased until we found enough neighbors. After
    # that we discard all neighbors except the k nearest.
    for multiplier in range(1, 11):
        if multiplier == 10:
            raise RuntimeError("Reached maximum radius")
        radii = [initial_radius * multiplier] * len(atoms)
        neighborhood = NeighborList(
            radii, skin=0.0, self_interaction=False, bothways=True
        )
        neighborhood.update(atoms)

        nodes = []
        dists = []
        edges = []
        senders = []
        receivers = []
        if np.any(atoms.get_pbc()):
            atom_positions = atoms.get_positions(wrap=True)
        else:
            atom_positions = atoms.get_positions(wrap=False)
        keep_edges = []
        keep_senders = []
        keep_receivers = []

        for i in range(len(atoms)):
            nodes.append(atom_numbers[i])

        early_exit = False
        for i in range(len(atoms)):
            this_dists = []
            this_edges = []
            this_senders = []
            this_receivers = []
            neighbor_indices, offset = neighborhood.get_neighbors(i)
            if len(neighbor_indices) < num_neighbors:
                # Not enough neigbors, so exit and increase radius
                early_exit = True
                break
            for j, offs in zip(neighbor_indices, offset):
                i_pos = atom_positions[i]
                j_pos = atom_positions[j] + np.dot(offs, unitcell)
                dist_vec = i_pos - j_pos
                dist = np.sqrt(np.dot(dist_vec, dist_vec))

                this_dists.append([dist])
                this_edges.append(dist_vec)
                this_senders.append(j)
                this_receivers.append(i)
            dists.append(np.array(this_dists))
            edges.append(np.array(this_edges))
            senders.append(np.array(this_senders))
            receivers.append(np.array(this_receivers))
        if early_exit:
            continue
        else:
            for d_ind, e_ind, s_ind, r_ind in zip(dists, edges, senders, receivers):
                # Keep only num_neighbors closest indices
                keep_ind = np.argsort(d_ind[:, 0])[0:num_neighbors]
                keep_edges.append(e_ind[keep_ind])
                keep_senders.append(s_ind[keep_ind])
                keep_receivers.append(r_ind[keep_ind])
        break
    return (
        np.array(nodes),
        atom_positions,
        np.concatenate(keep_edges).reshape(-1, 3),
        np.concatenate(keep_senders),
        np.concatenate(keep_receivers),
    )

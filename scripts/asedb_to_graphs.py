import numpy as np
import pandas
import ase.db
from ase import Atoms
from ase.neighborlist import NeighborList
#from ase.visualize import view

import sys
import argparse
from ast import literal_eval
import warnings


def get_graph_cutoff(atoms: Atoms, cutoff):
    '''Return the graph features, with cutoff adjacency.
    Inspired by https://github.com/peterbjorgensen/msgnet/blob/master/src/msgnet/dataloader.py'''

    nodes = [] # initialize arrays, to be filled in loop later
    senders = []
    receivers = []
    edges = []
    atom_numbers = atoms.get_atomic_numbers() # get array of atomic numbers

    # divide cutoff by 2, because ASE defines two atoms as neighbours 
    # when their spheres of radii r overlap
    radii = [cutoff/2] * len(atoms) # make a list with length len(atoms)
    neighborhood = NeighborList(
        radii, skin=0.0, self_interaction=False, bothways=True
    )
    neighborhood.update(atoms)

    if np.any(atoms.get_pbc()):
        atom_positions = atoms.get_positions(wrap=True)
    else:
        atom_positions = atoms.get_positions(wrap=False)

    unitcell = atoms.get_cell()
    
    for ii in range(len(atoms)):
            nodes.append(atom_numbers[ii])

    for ii in range(len(atoms)):
        neighbor_indices, offset = neighborhood.get_neighbors(ii)
        for jj, offs in zip(neighbor_indices, offset):
            ii_pos = atom_positions[ii]
            jj_pos = atom_positions[jj] + np.dot(offs, unitcell)
            dist_vec = ii_pos - jj_pos
            dist = np.sqrt(np.dot(dist_vec, dist_vec))

            senders.append(jj)
            receivers.append(ii)
            edges.append(dist)

    if len(edges) == 0:
        warnings.warn("Generated graph has zero edges")
        edges = np.zeros((0, 1))

    return (
        np.array(nodes),
        atom_positions,
        np.array(edges),
        np.array(senders),
        np.array(receivers)
    )

def get_graph_knearest(
    atoms: Atoms, num_neighbors, initial_radius=3.0
):
    '''Return the graph features, with knearest adjacency.
    Inspired by https://github.com/peterbjorgensen/msgnet/blob/master/src/msgnet/dataloader.py'''

    atoms.wrap() # put atoms inside unit cell by wrapping their positions
    atom_numbers = atoms.get_atomic_numbers()
    unitcell = atoms.get_cell()

    # We want to calculate k nearest neighbors, so we start within a sphere with radius R.
    # In this sphere we are calculating the number of neighbors, if there are not enough,
    # i.e. the number of neighbors within the sphere is smaller than k, R is increased 
    # until we found enough neighbors. After that we discard all neighbors except the k nearest. 
    for multiplier in range(1, 11):
        if multiplier == 10:
            raise RuntimeError("Reached maximum radius")
        radii = [initial_radius * multiplier] * len(atoms)
        neighborhood = NeighborList(
            radii, skin=0.0, self_interaction=False, bothways=True
        )
        neighborhood.update(atoms)

        nodes = []
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

        for ii in range(len(atoms)):
            nodes.append(atom_numbers[ii])

        early_exit = False
        for ii in range(len(atoms)):
            this_edges = []
            this_senders = []
            this_receivers = []
            neighbor_indices, offset = neighborhood.get_neighbors(ii)
            if len(neighbor_indices) < num_neighbors:
                # Not enough neigbors, so exit and increase radius
                early_exit = True
                break
            for jj, offs in zip(neighbor_indices, offset):
                ii_pos = atom_positions[ii]
                jj_pos = atom_positions[jj] + np.dot(offs, unitcell)
                dist_vec = ii_pos - jj_pos
                dist = np.sqrt(np.dot(dist_vec, dist_vec))

                this_edges.append([dist])
                this_senders.append(jj)
                this_receivers.append(ii)
            edges.append(np.array(this_edges))
            senders.append(np.array(this_senders))
            receivers.append(np.array(this_receivers))
        if early_exit:
            continue
        else:
            for e, s, r in zip(edges, senders, receivers):
                # Keep only num_neighbors closest indices
                keep_ind = np.argsort(e[:, 0])[0:num_neighbors]
                keep_edges.append(e[keep_ind])
                keep_senders.append(s[keep_ind])
                keep_receivers.append(r[keep_ind])
        break
    return (
        np.array(nodes),
        atom_positions,
        np.concatenate(keep_edges).flatten(),
        np.concatenate(keep_senders),
        np.concatenate(keep_receivers),
    )


def main(args):
    '''Load atoms and properties from ase database and convert them to graphs using a cutoff type,
    and store the graphs with properties in dataframe as .csv file.
    Calculating the graphs takes some time and should only be done once for each dataset.'''
    # needs parameters: cutoff type, cutoff dist, discard unconnected graphs
    db_in = ase.db.connect(args.file_in)
    graph_df = []
    with ase.db.connect(args.file_out, append=False) as db_out:

        for i, row in enumerate(db_in.select()):
            atoms = row.toatoms() # get the atoms object from each row
            prop_dict = row.key_value_pairs # get property dict

            # calculate adjacency of graph as senders and receivers
            cutoff = args.cutoff
            if args.cutoff_type == 'const':
                nodes, atom_positions, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)
            elif args.cutoff_type == 'knearest':
                cutoff = int(cutoff)
                nodes, atom_positions, edges, senders, receivers = get_graph_knearest(atoms, cutoff)
            else:
                raise ValueError(f'Cutoff type {args.cutoff_type} not recognised.')
            data = {'senders': senders, 'receivers': receivers, 'edges': edges}
            # add information about cutoff
            prop_dict['cutoff_type'] = args.cutoff_type
            prop_dict['cutoff_val'] = cutoff

            # save in new database
            db_out.write(atoms, key_value_pairs=prop_dict, data=data)
            
            if i<3:
                print(prop_dict)
            if i%1000 == 0:
                print(f'Step {i}')
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ase database to graphs in csv format.')
    parser.add_argument('-f', '-F', type=str, dest='file_in', required=True,
                        help='input file name')
    parser.add_argument('-o', type=str, dest='file_out', required=True,
                        help='output file name')
    parser.add_argument('-cutoff_type', type=str, dest='cutoff_type', required=True,
                        help='choose the cutoff type, knearest or const')            
    parser.add_argument('-cutoff', type=float, dest='cutoff', default=4.0,
                        help='cutoff distance or number of nearest neighbors')
    args = parser.parse_args()
    main(args)
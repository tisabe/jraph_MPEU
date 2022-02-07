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
    connections_offset = []
    edges = []
    atom_numbers = atoms.get_atomic_numbers() # get array of atomic numbers

    # divide cutoff by 2, because ASE defines two atoms as neighbours when their spheres of radii r overlap
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
            connections_offset.append(np.vstack((offs, np.zeros(3, float))))
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



def main(args):
    '''Load atoms and properties from ase database and convert them to graphs using a cutoff type,
    and store the graphs with properties in dataframe as .csv file.
    Calculating the graphs takes some time and should only be done once for each dataset.'''
    # needs parameters: cutoff type, cutoff dist, discard unconnected graphs
    db_in = ase.db.connect(args.file_in)
    graph_df = []
    db_out = ase.db.connect(args.file_out)

    for i, row in enumerate(db_in.select()):
        atoms = row.toatoms() # get the atoms object from each row
        prop_dict = row.key_value_pairs # get property dict

        # calculate adjacency of graph as senders and receivers
        cutoff = args.cutoff
        nodes, atom_positions, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)
        data = {'senders': senders, 'receivers': receivers, 'edges': edges}
        # add information about cutoff
        prop_dict['cutoff_type'] = 'const'
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
    parser.add_argument('-cutoff', type=str, dest='cutoff', default=4.0,
                        help='output file name')
    args = parser.parse_args()
    main(args)
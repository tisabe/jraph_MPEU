import numpy as np
import pandas
from ase import Atoms
#from ase.visualize import view
from ase.neighborlist import NeighborList
import sys

from ast import literal_eval
import warnings


def str_to_array(str_array):
    '''Return a numpy array converted from a single string, representing an array.'''
    return np.array(literal_eval(str_array))

def dict_to_ase(dict):
    '''Return ASE atoms object from a pandas dict, produced by AFLOW json response.'''
    cell = str_to_array(dict['geometry_orig'])
    positions = str_to_array(dict['positions_cartesian'])
    symbols = dict['compound']
    structure = Atoms(symbols=symbols,
                        positions=positions,
                        cell=cell,
                        pbc=(1,1,1))
    structure.wrap()
    return structure

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

def make_graph_df(aflow_df: pandas.DataFrame, cutoff):
    graph_df = []
    max_atoms = 64 # limit unitcell size to 64 atoms

    for index, row in aflow_df.iterrows():
        atoms = dict_to_ase(row)
        num_atoms = len(atoms)
        if num_atoms < max_atoms:
            auid = row['auid']
            #label = row['enthalpy_atom'] # change this for different target properties/labels
            label = row['Egap']
            nodes, atom_positions, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)
            graph = {
                'nodes' : nodes,
                'atom_positions' : atom_positions,
                'edges' : edges,
                'senders' : senders,
                'receivers' : receivers,
                'label' : label,
                'auid' : auid
            }
            graph_df.append(graph)
            
        if index%1000 == 0:
            print(index)
    
    graph_df = pandas.DataFrame(graph_df)
    return graph_df


def main():

    print('Starting datahandler')
    print('Starting aflow data to graphs conversion')
    # source file:
    np.set_printoptions(threshold=sys.maxsize) # there might be long arrays, so we have to prevent numpy from shortening them
    df_csv_file = 'aflow/aflow_binary_enthalpy_atom.csv'
    #df_csv_file = 'aflow/aflow_binary_egap_above_zero_below_ten_mill.csv'
    df = pandas.read_csv(df_csv_file)
    graph_df = make_graph_df(df, cutoff=4.0)
    print(graph_df.head())
    # target file:
    #graph_df.to_csv(('aflow/graphs_enthalpy_cutoff4A.csv'))
    graph_df.to_csv(('aflow/graphs_nonmetals.csv'))
    

if __name__ == "__main__":
    main()
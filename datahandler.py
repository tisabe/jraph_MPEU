import numpy as np
import pandas
from ase import Atoms
#from ase.visualize import view
from ase.neighborlist import NeighborList

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

def make_graph_df(aflow_df: pandas.DataFrame):
    graph_df = []

    for index, row in aflow_df.iterrows():
        atoms = dict_to_ase(row)
        label = row['Egap'] # change this for different target properties/labels
        nodes, atom_positions, edges, senders, receivers = get_graph_cutoff(atoms, 3.0)
        graph = {
            'nodes' : nodes,
            'atom_positions' : atom_positions,
            'edges' : edges,
            'senders' : senders,
            'receivers' : receivers,
            'label' : label
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
    df_csv_file = 'aflow/aflow_binary_egap_above_zero_below_ten_mill.csv'
    df = pandas.read_csv(df_csv_file)
    graph_df = make_graph_df(df)
    print(graph_df.head())
    # target file:
    graph_df.to_csv(('aflow/graphs_test_cutoff3A.csv'))
    

if __name__ == "__main__":
    main()
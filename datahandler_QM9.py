import numpy as np
import pandas
from ase.visualize import view
from ase import Atoms
import spektral
from spektral.datasets import QM9
import sys

from datahandler import get_graph_cutoff

species_QM9_dict = np.array(['H', 'C', 'N', 'O', 'F'])

def spektral_to_ase(graph_s: spektral.data.graph.Graph) -> Atoms:
    nodes = graph_s.x
    atomic_indices = np.array(np.argmax(nodes[:,0:5], axis=1))
    species = species_QM9_dict[atomic_indices]
    #print(nodes[:,0:5])
    #print(species)
    cell = [1, 1, 1]
    atom_positions = nodes[:, 5:8]
    struct = Atoms(species,
                    positions = atom_positions,
                    cell = cell,
                    pbc = [0,0,0])
    return struct

def get_QM9_label(graph_s: spektral.data.graph.Graph, index):
    label = graph_s.y[index]
    return label

def make_QM9_df(data, cutoff, index):
    graph_df = []
    
    for i, graph_s in enumerate(data):
        max_atoms = 64
        atoms = spektral_to_ase(graph_s)
        num_atoms = len(atoms)
        if num_atoms < max_atoms:
            #label = row['enthalpy_atom'] # change this for different target properties/labels
            label = get_QM9_label(graph_s, index)
            nodes, atom_positions, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)
            graph = {
                'nodes' : nodes,
                'atom_positions' : atom_positions,
                'edges' : edges,
                'senders' : senders,
                'receivers' : receivers,
                'label' : label,
                'auid' : i
            }
            graph_df.append(graph)
            
        if i%1000 == 0:
            print(i)
    
    graph_df = pandas.DataFrame(graph_df)
    return graph_df

def test():
    dataset = QM9(amount=1024)

    graph_s = dataset[1]
    #print(graph_s)
    atoms = spektral_to_ase(graph_s)
    #view(atoms)
    cutoff = 4
    nodes, atom_positions, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)
    print(nodes)
    print(atom_positions)
    print(edges)
    print(senders)
    print(receivers)
    print(get_QM9_label(graph_s, 7))
    #view(atoms)
    df = make_QM9_df(dataset, cutoff=4, index=7)
    print(df.loc[1])

def main():

    dataset = QM9(amount = 1024*16)
    graph_s = dataset[1]
    print(graph_s.y)
    np.set_printoptions(threshold=sys.maxsize) # there might be long arrays, so we have to prevent numpy from shortening them
    #df_csv_file = 'aflow/aflow_binary_enthalpy_atom.csv'
    graph_df = make_QM9_df(dataset, cutoff=4.0, index=16)
    print(graph_df.head())
    # target file:
    #graph_df.to_csv(('aflow/graphs_enthalpy_cutoff4A.csv'))
    graph_df.to_csv(('QM9/graphs_U0K.csv'))
    return 0

if __name__ == "__main__":
    main()
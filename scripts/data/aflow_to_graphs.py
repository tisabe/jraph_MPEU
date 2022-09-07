import numpy as np
import pandas
import ase
from ase import Atoms
from ase.visualize import view

from ast import literal_eval
import argparse

from asedb_to_graphs import (
    get_graph_fc,
    get_graph_cutoff,
    get_graph_knearest
)


def str_to_array(str_array):
    '''Return a numpy array converted from a single string, representing an array.'''
    return np.array(literal_eval(str_array))

def dict_to_ase(aflow_dict):
    '''Return ASE atoms object from a pandas dict, produced by AFLOW json response.'''
    cell = str_to_array(aflow_dict['geometry_orig'])
    positions = str_to_array(aflow_dict['positions_fractional'])
    symbols = aflow_dict['compound']
    structure = Atoms(
        symbols=symbols,
        scaled_positions=positions,
        cell=cell,
        pbc=(1, 1, 1))
    structure.wrap()
    return structure


def main(args):
    """Load aflow data from csv file into ase database with graph features,
    i.e. edges and adjacency list."""
    # needs parameters: cutoff type, cutoff dist, discard unconnected graphs
    aflow_df = pandas.read_csv(args.file_in, index_col=0)
    with ase.db.connect(args.file_out, append=False) as db_out:
        for i, row in aflow_df.iterrows():
            atoms = dict_to_ase(row)  # get the atoms object from each row
            row = row.to_dict()
            row.pop('geometry_orig', None)
            row.pop('positions_fractional', None)
            row.pop('compound', None)

            # calculate adjacency of graph as senders and receivers
            cutoff = args.cutoff
            if args.cutoff_type == 'const':
                nodes, atom_positions, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)
            elif args.cutoff_type == 'knearest':
                cutoff = int(cutoff)
                nodes, atom_positions, edges, senders, receivers = get_graph_knearest(atoms, cutoff)
            elif args.cutoff_type == 'fc':
                nodes, atom_positions, edges, senders, receivers = get_graph_fc(atoms)
            else:
                raise ValueError(f'Cutoff type {args.cutoff_type} not recognised.')

            # get property dict from all keys in the row
            prop_dict = {}
            data = {}
            for key in row.keys():
                val = row[key]
                prop_dict[key] = val
            data['senders'] = senders
            data['receivers'] = receivers
            data['edges'] = edges
            # add information about cutoff
            prop_dict['cutoff_type'] = args.cutoff_type
            prop_dict['cutoff_val'] = cutoff

            # save in new database
            db_out.write(atoms, key_value_pairs=prop_dict, data=data)

            if i<3:
                print(prop_dict)
                print(atoms)
                view(atoms)
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

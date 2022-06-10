import numpy as np
import pandas
import ase.db

import sys
import argparse

from jraph_MPEU.input_pipeline import (
    get_graph_cutoff,
    get_graph_knearest,
    get_graph_fc
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
            elif args.cutoff_type == 'fc':
                nodes, atom_positions, edges, senders, receivers = get_graph_fc(atoms)
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
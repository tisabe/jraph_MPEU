import numpy as np
import pandas
import ase.db
from ase import Atoms
#from ase.visualize import view
from ase.neighborlist import NeighborList

import sys
import argparse
from ast import literal_eval
import warnings



def main(args):
    '''Load atoms and properties from ase database and convert them to graphs using a cutoff type,
    and store the graphs with properties in dataframe as .csv file.
    Calculating the graphs takes some time and should only be done once for each dataset.'''
    # needs parameters: cutoff type, cutoff dist, discard unconnected graphs
    db = ase.db.connect(args.file_in)
    graph_df = []
    for i, row in enumerate(db.select()):
        atoms = row.toatoms() # get the atoms object from each row
        prop_dict = row.key_value_pairs # get property dict
        print(prop_dict)
        if i>3:
            break
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ase database to graphs in csv format.')
    parser.add_argument('-f', '-F', type=str, dest='file_in', required=True,
                        help='input file name')
    parser.add_argument('-o', type=str, dest='file_out', required=True,
                        help='output file name')
    args = parser.parse_args()
    main(args)
import numpy as np
import pandas
from ase import Atoms
#from ase.visualize import view
from ase.neighborlist import NeighborList
import sys

from ast import literal_eval
import warnings

from datahandler import *

# source file:
np.set_printoptions(threshold=sys.maxsize) # there might be long arrays, so we have to prevent numpy from shortening them
df_csv_file = 'aflow/aflow_binary_egap_above_zero_below_ten_mill.csv'
df = pandas.read_csv(df_csv_file)

row = df.iloc[0]
atoms = dict_to_ase(row)
print(atoms)

nodes, pos, edges, adj, adj_offset = get_graph_knearest(atoms, 4)
print('Nodes:')
print(nodes)
print('Edges:')
print(edges)
print('Adj:')
print(adj)
print('Adj offset:')
print(adj_offset)




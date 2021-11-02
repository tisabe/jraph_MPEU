import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
from utils import *
from model import get_data_df_csv

folder_data = 'aflow/'
file_graphs = 'graphs_enthalpy_cutoff4A.csv'

df_graphs = pandas.read_csv(folder_data+file_graphs)
inputs, outputs, auids = get_data_df_csv(folder_data+file_graphs)

#print(inputs[0])
#print(inputs[-1])
num_edges = []
num_nodes = []

for graph in inputs:
    num_edges.append(graph.n_edge[0])
    num_nodes.append(graph.n_node[0])

num_edges = np.array(num_edges)
num_nodes = np.array(num_nodes)
max_edges = max(num_edges)
max_nodes = max(num_nodes)
print(num_edges)
print(num_nodes)

fig, ax = plt.subplots(2,1)

ax[0].hist(num_edges, bins=max_edges)
ax[0].set_yscale('log')

ax[1].hist(num_nodes, bins=max_nodes)
ax[1].set_yscale('log')

plt.show()













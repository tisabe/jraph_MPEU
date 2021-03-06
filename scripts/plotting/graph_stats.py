import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
from utils import *

folder_data = 'QM9/'
#file_graphs = 'graphs_all_labelidx16.csv'
file_graphs = 'graphs_16k.csv'

label_str = 'gap'
inputs, outputs, auids = get_data_df_csv(folder_data+file_graphs, label_str=label_str)

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
print('Total number of edges: {}'.format(np.sum(num_edges)))
print('Total number of nodes: {}'.format(np.sum(num_nodes)))

fig, ax = plt.subplots(2,1)

ax[0].hist(num_edges, bins=max_edges)
ax[0].set_yscale('log')
ax[0].set_xlabel('number of edges per graph')
ax[0].set_ylabel('count')

ax[1].hist(num_nodes, bins=max_nodes)
ax[1].set_yscale('log')
ax[1].set_xlabel('number of nodes per graph')
ax[1].set_ylabel('count')

plt.show()

# plot label value depending on number of atoms
fig, ax = plt.subplots()

ax.scatter(num_nodes, outputs)
ax.set_xlabel('number of atoms')
ax.set_ylabel(label_str)

plt.show()













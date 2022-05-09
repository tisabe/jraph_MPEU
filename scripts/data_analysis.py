import argparse
from pathlib import Path

import numpy as np
import ase.db
import pandas as pd

import matplotlib.pyplot as plt



def main(args):
    file = args.file
    folder = file.replace('.db', '')  # make a name for the plot output folder
    folder = 'scripts/plots/'+folder
    Path(folder).mkdir(parents=True, exist_ok=True)
    limit = args.limit
    key = args.key

    num_nodes = []
    num_senders = []
    num_receivers = []
    num_edges = []
    edges_all = np.array([]) # collect all edge distances for histogram
    key_val_list = [] # list of key-value-pairs

    with ase.db.connect(file) as asedb:
        for i, row in enumerate(asedb.select(limit=limit)):
            if i%10000 == 0:
                print(f'Reading step {i}')
            key_value_pairs = row.key_value_pairs
            key_val_list.append(key_value_pairs)
            data = row.data
            num_nodes.append(row.natoms)
            #senders = data['senders']
            #receivers = data['receivers']
            edges = data['edges']
            num_edges.append(len(edges))
            #edges_all = np.concatenate((edges_all, np.array(edges)))


    key_df = pd.DataFrame(key_val_list)
    print(key_df.head())
    print(key_df.describe())
    key_y = key_df.get(key).to_numpy()
    units = input("Type units of key: ")
    fig, ax = plt.subplots()
    ax.hist(key_y, bins=100, log=True)
    ax.set_xlabel(f'{key} ({units})', fontsize=12)
    ax.set_ylabel('Number of graphs', fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(folder+f'/{key}_hist.png', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots(2, 1)
    ax[0].hist(num_nodes, bins=100, log=True)
    ax[0].set_xlabel('Number of nodes', fontsize=12)
    ax[1].hist(num_edges, bins=100, log=True)
    ax[1].set_xlabel('Number of edges', fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(folder+'/graph_stat_hist.png', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    ax.scatter(num_nodes, num_edges)
    ax.set_xlabel('Number of nodes', fontsize=12)
    ax.set_ylabel('Number of edges', fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(folder+'/edges_per_node.png', bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show data analysis plots.')
    parser.add_argument('-f', '-F', type=str, dest='file', default='QM9/qm9_graphs.db',
                        help='data directory name')
    parser.add_argument('-limit', type=int, dest='limit', default=None,
                        help='limit number of database entries to be selected')
    parser.add_argument('-key', type=str, dest='key', default=None,
                        help='key name to plot')
    args = parser.parse_args()
    main(args)

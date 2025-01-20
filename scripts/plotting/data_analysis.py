"""This script plots various metrics to investigate the databases."""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import ase.db
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns

# We define a blacklist for keys that should be excluded from being plotted.
# This can be extended if additional data is pulled.
keys_blacklist = [
    'auid',
    'aurl',
    'tag',
    'index',
    'cutoff_type',
    'cutoff_val',
    'mp-id'
]

AFLOW_DB = '/home/dts/Documents/hu/batch_stats/graphs_knn_for_histogram.db'
qm9_DB = '/home/dts/Documents/hu/batch_stats/qm9_graphs_fc.db'
DB_LIST = [AFLOW_DB, qm9_DB]
# DB_LIST = [qm9_DB]

DB_LABEL_LIST = ['AFLOW', 'QM9']

FONTSIZE = 16
# FONT = 'Times'
# FONT = 'Times new roman'
FONT = 'serif'
ticksize=12


def create_histogram_of_datasets(limit=None):
    """Main function for database connection and plotting."""
    # file = args.file
    # folder = file.replace('.db', '')  # make a name for the plot output folder
    # folder = 'scripts/figs/'+folder
    # Path(folder).mkdir(parents=True, exist_ok=True)
    # limit = args.limit
    # key = args.key

    # bins_bottom = 800
    # bins_top = 50

    # x_lims = [0, 32]
    # ax[1].text(0.1, 0.1, 'dynamic', fontsize=32)   
    # batch size 32, qm9
    x1_lims = [0, 100]
    x2_lims = [0, 1000]
    # bins_top=np.arange(x1_lims[0], x1_lims[1], step=0.5)+0.5 
    bins_top=np.arange(x1_lims[0], x1_lims[1], step=1)+0.5 

    # bins_bottom=np.arange(x2_lims[0], x2_lims[1], step=20)+0.5 
    bins_bottom=np.arange(x2_lims[0], x2_lims[1], step=10)+0.5 

    plt.rc('xtick', labelsize=FONTSIZE-2)
    plt.rc('ytick', labelsize=FONTSIZE-2)
    plt.rc('legend', fontsize=FONTSIZE)
    plt.rc('legend', title_fontsize=FONTSIZE)
    plt.rc('axes', labelsize=FONTSIZE)

    fig, ax = plt.subplots(2, 1, figsize=(5.1, 5.1))


    for db_label, database in zip(DB_LABEL_LIST, DB_LIST):
        num_nodes = []
        num_edges = []
        atomic_numbers_all = np.array([])
        edges_all = [] # collect all edge distances for histogram
        key_val_list = [] # list of key-value-pairs
        with ase.db.connect(database) as asedb:
            for i, row in enumerate(asedb.select(limit=limit)):
                if i % 10000 == 0:
                    print(f'Reading step {i}')
                key_value_pairs = row.key_value_pairs
                key_val_list.append(key_value_pairs)
                data = row.data
                atomic_numbers = row.numbers
                atomic_numbers_all = np.concatenate(
                    (atomic_numbers_all, atomic_numbers), axis=None)
                num_nodes.append(int(row.natoms))
                #senders = data['senders']
                #receivers = data['receivers']
                edges = data['edges']
                num_edges.append(len(edges))
                edges_all.append(np.array(edges))
        print(f'Database: {db_label}')
        # print(f'Database location: {database}')
        # print(edges_all[0:6])
        # print('concatenate')
        edges_all = np.concatenate(edges_all, axis=0)
        # print(edges_all[0:6])
        # print('np lin alg norm')
        dists = np.linalg.norm(edges_all, axis=1)

        print(
            f'dataset: {database}, has this many num_nodes entries: {len(num_nodes)}'
            f' and this many num_edges: {len(num_edges)}')
        # Plot the histogram of nodes
        ax[0].hist(
            num_nodes, bins_top, density=False, histtype='bar',
            alpha=0.5, label=db_label)


        # Now plot the histogram of edges
        ax[1].hist(
            num_edges, bins_bottom, density=False, histtype='bar', alpha=0.5, label=db_label)


        import matplotlib.font_manager as font_manager
        font = font_manager.FontProperties(family=FONT,
                                        # weight='bold',
                                        style='normal', size=12)
    ax[1].legend(loc='upper right', prop=font, edgecolor="black", fancybox=False)
    ax[1].set_xlabel('Number of edges', fontsize=FONTSIZE, font=FONT)
    ax[1].set_ylabel('Count', fontsize=FONTSIZE, font=FONT)

    ax[0].set_xlabel('Number of nodes', fontsize=FONTSIZE, font=FONT)
    ax[0].set_ylabel('Count', fontsize=FONTSIZE, font=FONT)
    # ax[0].set_xlim(0, 100)
    plt.tight_layout()
    fig.align_labels()

    plt.show()
    
    fig.savefig(
        '/home/dts/Documents/theory/batching_paper/figs/histogram_of_datasets.png',
        bbox_inches='tight', dpi=600)

        # fig, ax = plt.subplots()
        # ax.hist(atomic_numbers_all, bins=int(max(atomic_numbers_all))+1, log=True)
        # ax.set_xlabel('Atomic number')
        # ax.set_ylabel('Number of nodes')
        # plt.tight_layout()
        # plt.show()

        # key_df = pd.DataFrame(key_val_list)
        # print(key_df.head())
        # print(key_df.describe())
        # for key_i in key_df.keys():
        #     if not key_i in keys_blacklist:
        #         print(key_i)

        #         key_y = key_df.get(key_i).to_numpy()
        #         units = ''#input("Type units of key: ")
        #         fig, ax = plt.subplots()
        #         if isinstance(key_y[0], float):
        #             ax.hist(key_y, bins=100, log=True)
        #         elif isinstance(key_y[0], str):
        #             sns.histplot(ax=ax, data=key_df, x=key_i, discrete=True)
        #             plt.xticks(rotation=90)
        #             plt.yscale('log')
        #         else:
        #             sns.histplot(ax=ax, data=key_df, x=key_i, discrete=True)
        #             plt.yscale('log')
        #         ax.set_xlabel(f'{key_i} ({units})')
        #         ax.set_ylabel('Number of graphs')
        #         plt.tight_layout()
        #         plt.show()
        #         fig.savefig(folder+f'/{key_i}_hist.png', bbox_inches='tight', dpi=600)

        # fig, ax = plt.subplots(2, 1)
        # ax[0].hist(num_nodes, bins=20, log=False)
        # ax[0].set_xlabel('Number of nodes')
        # ax[0].set_ylabel('Count')
        # ax[1].hist(num_edges, bins=20, log=False)
        # ax[1].set_xlabel('Number of edges')
        # ax[1].set_ylabel('Count')
        # plt.tight_layout()
        # plt.show()
        # fig.savefig(folder+'/graph_stat_hist.png', bbox_inches='tight', dpi=600)

        # fig, ax = plt.subplots()
        # ax.scatter(num_nodes, num_edges)
        # ax.set_xlabel('Number of nodes')
        # ax.set_ylabel('Number of edges')
        # plt.tight_layout()
        # plt.show()
        # fig.savefig(folder+'/edges_per_node.png', bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Show data analysis plots.')
    # parser.add_argument('-f', '-F', type=str, dest='file',
    #                     default='databases/QM9/graphs_fc_vec.db',
    #                     help='data directory name')
    # parser.add_argument('-limit', type=int, dest='limit', default=None,
    #                     help='limit number of database entries to be selected')
    # parser.add_argument('-key', type=str, dest='key', default=None,
    #                     help='key name to plot')
    # args_main = parser.parse_args()
    # main(args_main)
    create_histogram_of_datasets()

"""This script plots various metrics to investigate the databases."""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import ase.db
import pandas as pd
from absl import app
import matplotlib.pyplot as plt
import scienceplots
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
node_and_edge_csv_aflow = '/home/dts/Documents/hu/batch_stats/aflow_node_edge_distribution_df.csv'
node_and_edge_csv_qm9 = '/home/dts/Documents/hu/batch_stats/qm9_node_edge_distribution_df.csv'

DB_LIST = [AFLOW_DB, qm9_DB]
# DB_LIST = [qm9_DB]

DB_LABEL_LIST = ['AFLOW', 'QM9']

FONTSIZE = 12
# FONT = 'Times'
# FONT = 'Times new roman'
FONT = 'serif'
ticksize=12


def get_histogram_data_from_db(database_path, limit=None):
    """Gets histogram data about nodes/edges distribution for an ASE db."""
    num_nodes = []
    num_edges = []
    atomic_numbers_all = np.array([])
    edges_all = [] # collect all edge distances for histogram
    key_val_list = [] # list of key-value-pairs
    with ase.db.connect(database_path) as asedb:
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
    print(f'Database: {database_path}')

    edges_all = np.concatenate(edges_all, axis=0)

    dists = np.linalg.norm(edges_all, axis=1)
    return num_nodes, num_edges


def create_histogram_of_datasets(aflow_nodes, aflow_edges, qm9_nodes, qm9_edges):
    """Main function for database connection and plotting."""
    density = True
    x1_lims = [0, 100]
    x2_lims = [0, 1000]
    bins_top=np.arange(x1_lims[0], x1_lims[1], step=1)+0.5 
    bins_bottom=np.arange(x2_lims[0], x2_lims[1], step=10)+0.5 
    # plt.rc('xtick', labelsize=FONTSIZE-2)
    # plt.rc('ytick', labelsize=FONTSIZE-2)
    # plt.rc('legend', fontsize=FONTSIZE)
    # plt.rc('legend', title_fontsize=FONTSIZE)
    # plt.rc('axes', labelsize=FONTSIZE)
    fig, ax = plt.subplots(2, 1, figsize=(5.1, 5.6))


    ax[0].hist(
        aflow_nodes, bins_top, density=density, histtype='bar',
        alpha=0.5, label='AFLOW')
    # Now plot the histogram of edges
    ax[1].hist(
        aflow_edges, bins_bottom, density=density, histtype='bar', alpha=0.5, label='AFLOW')
    
    ax[0].hist(
        qm9_nodes, bins_top, density=density, histtype='bar',
        alpha=0.5, label='QM9')
    # Now plot the histogram of edges
    ax[1].hist(
        qm9_edges, bins_bottom, density=density, histtype='bar', alpha=0.5, label='QM9')

    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)
    # for i in range(2):
    #     ax[i].tick_params(axis='both', which='minor', labelsize=FONTSIZE-2)
    #     ax[i].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    #     plt.setp(ax[i].get_xticklabels(), fontsize=FONTSIZE-2, font=FONT) 
    #     plt.setp(ax[i].get_yticklabels(), fontsize=FONTSIZE-2, font=FONT)
    ax[0].set_xticklabels([0, 20, 40, 60, 80, 100], fontsize=FONTSIZE, font=FONT)
    ax[0].set_xticks([0, 20, 40, 60, 80, 100], fontsize=FONTSIZE, font=FONT)
    ax[0].set_xlim([0, 100])
    ax[0].set_ylim([0, 0.8])
    ax[0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8], fontsize=FONTSIZE, font=FONT)
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=FONTSIZE, font=FONT)

    ax[1].set_xticklabels([0, 200, 400, 600, 800, 1000], fontsize=FONTSIZE, font=FONT)
    ax[1].set_xticks([0, 200, 400, 600, 800, 1000], fontsize=FONTSIZE, font=FONT)
    ax[1].legend(loc='upper right', prop=font, edgecolor="black", fancybox=False)
    ax[1].set_xlim([0, 1000])
    ax[1].set_ylim([0, 0.08])
    ax[1].set_yticklabels([0, 0.02, 0.04, 0.06, 0.08], fontsize=FONTSIZE, font=FONT)
    ax[1].set_yticks([0, 0.02, 0.04, 0.06, 0.08], fontsize=FONTSIZE, font=FONT)
    ax[1].set_xlabel('Number of edges', fontsize=FONTSIZE, font=FONT)
    ax[1].set_ylabel('Density', fontsize=FONTSIZE, font=FONT)

    ax[0].set_xlabel('Number of nodes', fontsize=FONTSIZE, font=FONT)
    ax[0].set_ylabel('Density', fontsize=FONTSIZE, font=FONT)
    plt.style.use(["science", "grid"])

    plt.tight_layout()
    fig.align_labels()

    plt.show()
    
    fig.savefig(
        '/home/dts/Documents/theory/batching_paper/figs/histogram_of_datasets.png',
        bbox_inches='tight', dpi=800)


def main(args):
    # Get qm9 and aflow data.
    # for db_label, database in zip(DB_LABEL_LIST, DB_LIST):
    #     print(f'getting data from {db_label} at path {database}')
    
    # qm9_nodes, qm9_edges = get_histogram_data_from_db(DB_LIST[1]) 
    # aflow_nodes, aflow_edges = get_histogram_data_from_db(DB_LIST[0])
    # aflow_df = pd.DataFrame({
    #     'aflow_nodes': aflow_nodes,
    #     'aflow_edges': aflow_edges,
    # })
    # aflow_df.to_csv(node_and_edge_csv_aflow, index=False)
    # qm9_df = pd.DataFrame({
    #     'qm9_nodes': qm9_nodes,
    #     'qm9_edges': qm9_edges,
    # })
    # qm9_df.to_csv(node_and_edge_csv_qm9, index=False)

    # Now read in the data from the csv.
    qm9_df = pd.read_csv(node_and_edge_csv_qm9)
    aflow_df = pd.read_csv(node_and_edge_csv_aflow)

    create_histogram_of_datasets(aflow_df['aflow_nodes'], aflow_df['aflow_edges'], qm9_df['qm9_nodes'], qm9_df['qm9_edges'])

if __name__ == '__main__':
    app.run(main)
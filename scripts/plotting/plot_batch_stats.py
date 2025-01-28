"""In this script we want to create histograms about batches.

For each dataset (AFLOW/qm9) we want to have histogram

Count on the left
Right axis is the number of nodes pre batching

Ideally plot the batch size 16, 32, 64 and 128 on top of this plot.

Ideally a subplot (2,1) where dynamic batching is on bottom, static on top.

We have a list of simulations we ran, but im not sure we need them.
"""
from absl import app
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# BATCH_SIZE_LIST = [16, 32, 64, 128]
# COLORS_LIST = ['red', 'tan', 'lime', 'black']

BATCH_SIZE_LIST = [16, 32, 64]
COLORS_LIST = ['red']

# BATCH_METHOD_LIST = ['static', 'dynamic']
BATCH_METHOD_LIST = ['dynamic', 'static']
N_BINS = 100
DATASET_LIST = ['AFLOW', 'qm9']

FONTSIZE = 12
FONT = 'serif'
ticksize=12




def create_histogram_num_graphs_dynamic(
        base_dir, model_type='MPEU', batch_stats_col='num_graph_before_batching', dataset='aflow'):
    batching_round_to_64 = False
    batch_size = 32
    dataset = 'aflow'
    # dataset = 'qm9'

    model_type = 'MPEU'
    # fig, ax = plt.subplots(2, 1, figsize=(6.472, 4), sharex=True, sharey=True)
    fig, ax = plt.subplots(1, 1, figsize=(5.1, 4))
    x_lims = [0, 32]
    # batch size 32, qm9
    # I think I should use step = 1!!!!
    bins=np.arange(x_lims[0], x_lims[1], step=1)+0.5               

    stats_array = get_stats_for_single_batch_size(base_dir,
        batch_stats_col, 'dynamic', dataset, batching_round_to_64,
        batch_size, model_type)
    ax.hist(
        stats_array, bins, density=True, histtype='bar', label='AFLOW', alpha=0.5)

    # Now add the QM9 data:
    dataset = 'qm9'
    bins=np.arange(x_lims[0], x_lims[1], step=1)+0.5               

    stats_array = get_stats_for_single_batch_size(base_dir,
        batch_stats_col, 'dynamic', dataset, batching_round_to_64,
        batch_size, model_type)
    ax.hist(
        stats_array, bins, density=True, histtype='bar', label='QM9', alpha=0.5)

    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)

    ax.legend(loc='upper left', prop=font, edgecolor="black", fancybox=False)

    ax.set_xlabel('Graphs in batch before padding', fontsize=FONTSIZE, font=FONT)
    ax.set_ylabel('Density', fontsize=FONTSIZE, font=FONT)

    ax.set_xlim(x_lims)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35], font=FONT, fontsize=FONTSIZE)

    ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35], font=FONT, fontsize=FONTSIZE)
    ax.set_ylim([0, 0.7])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], font=FONT, fontsize=FONTSIZE)

    ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], font=FONT, fontsize=FONTSIZE)
    
    plt.tight_layout()
    plt.savefig('/home/dts/Documents/theory/batching_paper/figs/batch_stats_num_graphs_before_batching_size_32_aflow.png', dpi=600)
    plt.show()


def create_histogram(
        base_dir, model_type='MPEU',
        batch_stats_col='num_edge_before_batching',
        dataset='aflow',
        x_lims=(500,620),
        y_lims=(0, 0.05),
        batch_size = 32
        ):
    batching_round_to_64 = False
    fig, ax = plt.subplots(2, 1, figsize=(5.1, 5.5), sharex=True, sharey=True)

    bins=np.arange(x_lims[0],x_lims[1])+0.5

    stats_array = get_stats_for_single_batch_size(base_dir,
        batch_stats_col, 'static', dataset, batching_round_to_64,
        batch_size, model_type)
    ax[0].hist(
        stats_array, bins, density=True, histtype='bar')
              

    stats_array = get_stats_for_single_batch_size(base_dir,
        batch_stats_col, 'dynamic', dataset, batching_round_to_64,
        batch_size, model_type)
    ax[1].hist(
        stats_array, bins, density=True, histtype='bar')
    ax[1].set_xlabel(f'Number of {batch_stats_col.split("_")[1]}s in batch before padding',
                     fontsize=12, font=FONT)
    ax[0].set_ylabel('Density', fontsize=FONTSIZE, font=FONT)
    ax[1].set_ylabel('Density', fontsize=FONTSIZE, font=FONT)

    ax[1].set_xlim([x_lims[0], x_lims[1]])    
    ax[0].set_ylim([y_lims[0], y_lims[1]])               
    ax[1].set_ylim([y_lims[0], y_lims[1]])     

    plt.tight_layout()
    plt.savefig(f'/home/dts/Documents/theory/batching_paper/figs/batch_stats_{batch_stats_col}_size_{str(batch_size)}_{dataset}.png', dpi=600)
    plt.show()


def create_nodes_and_edges_histogram(base_dir, model_type='MPEU',
        dataset='aflow',
        x1_lims=(500,620),
        x2_lims=(8000,12000),
        y1_lims=(0, 0.06),
        y2_lims=(0, 0.003),
        batch_size=32):
    batching_round_to_64 = False

    # Column names to pull
    num_edges_col= 'num_edge_before_batching'
    num_nodes_col= 'num_node_before_batching'

    fig, ax = plt.subplots(2, 1, figsize=(5.1, 5.6), sharex=False, sharey=False)

    bins_lhs = np.arange(x1_lims[0], x1_lims[1])+0.5

    bins_rhs = 120 # np.arange(x2_lims[0], x2_lims[1])+0.5

    node_stats_array_static = get_stats_for_single_batch_size(base_dir,
        num_nodes_col, 'static', dataset, batching_round_to_64,
        batch_size, model_type)
    node_stats_array_dynamic = get_stats_for_single_batch_size(base_dir,
        num_nodes_col, 'dynamic', dataset, batching_round_to_64,
        batch_size, model_type)

    ax[0].hist(
        node_stats_array_dynamic, bins_lhs,
        density=True, histtype='bar', alpha=0.5, label='dynamic')
    ax[0].hist(
        node_stats_array_static, bins_lhs,
        density=True, histtype='bar', alpha=0.5, label='static')

    edge_stats_array_static = get_stats_for_single_batch_size(base_dir,
        num_edges_col, 'static', dataset, batching_round_to_64,
        batch_size, model_type)
    edge_stats_array_dynamic = get_stats_for_single_batch_size(base_dir,
        num_edges_col, 'dynamic', dataset, batching_round_to_64,
        batch_size, model_type)


    ax[1].hist(
        edge_stats_array_dynamic, bins_rhs, density=True,
        histtype='bar', alpha=0.5, label='dynamic')
    ax[1].hist(
        edge_stats_array_static, bins_rhs, density=True,
        histtype='bar', alpha=0.5, label='static')

    
    ax[0].set_xlabel(f'Nodes in batch before padding',
                     fontsize=FONTSIZE, font=FONT)
    ax[1].set_xlabel(f'Edges in batch before padding',
                     fontsize=FONTSIZE, font=FONT)
    ax[0].set_ylabel('Density', fontsize=FONTSIZE, font=FONT)
    ax[1].set_ylabel('Density', fontsize=FONTSIZE, font=FONT)
    ax[0].set_xticklabels([500, 520, 540, 560, 580, 600, 620], font=FONT, fontsize=FONTSIZE)
    ax[0].set_yticks([0, 0.02, 0.04, 0.06], font=FONT, fontsize=FONTSIZE)

    ax[0].set_yticklabels([0, 0.02, 0.04, 0.06], font=FONT, fontsize=FONTSIZE)

    ax[1].set_xticklabels([8000, 9000, 10000, 11000, 12000], font=FONT, fontsize=FONTSIZE)
    ax[1].set_yticks([0, 0.001, 0.002, 0.003])
    ax[1].set_yticklabels([0, 0.001, 0.002, 0.003], font=FONT, fontsize=FONTSIZE)

    ax[0].set_xlim([x1_lims[0], x1_lims[1]])

    ax[1].set_xlim([x2_lims[0], x2_lims[1]])
    
    ax[0].set_ylim([y1_lims[0], y1_lims[1]])                    
    ax[1].set_ylim([y2_lims[0], y2_lims[1]])                    
    ax[1].set_xticks([8000, 9000, 10000, 11000, 12000])
    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)

    ax[1].legend(loc='upper right', prop=font, edgecolor="black", fancybox=False)
    fig.align_labels()
    plt.tight_layout()
    plt.savefig(f'/home/dts/Documents/theory/batching_paper/figs/batch_stats_nodes_edges_size_{str(batch_size)}_{dataset}.png', dpi=600)
    plt.show()



def get_batch_stats_data(
        base_dir, batch_stats_col, batching_method, dataset, batching_round_to_64,
        model_type):
    """Should return an numpy array with batch sttas for each batch size.
    
    The returned dimensions should be 10k (steps) vs 4 (different batch sizes).
    """
    batch_stats_data = []
    for batch_size in BATCH_SIZE_LIST:
        stats_list = get_stats_for_single_batch_size(
            base_dir, batch_stats_col, batching_method, dataset, batching_round_to_64,
            batch_size, model_type)
        batch_stats_data.append(stats_list)
        # Combine these batch data to a single numpy array.
    return np.transpose(np.array(batch_stats_data))


def get_stats_for_single_batch_size(base_dir,
        batch_stats_col, batching_method, dataset, batching_round_to_64,
        batch_size, model_type):
    # For a single batch size
    csv_filename = get_csv_name_single_batch(
        base_dir, model_type, batching_method, dataset, batching_round_to_64,
        batch_size)
    # Read the csv
    df = pd.read_csv(csv_filename)\
    # Now grab the column we want:
    return df[batch_stats_col]


def get_csv_name_single_batch(
        base_dir, model_type, batching_method, dataset, batching_round_to_64,
        batch_size):
    """Here's an example of a file:
    
    ~/batching_both_models_both_datasets_100k_steps_21_3_2024/profiling_experiments/MPEU/qm9/dynamic/round_True/32/gpu_a100/iteration_0
    
    """
    # Create a file path from the list of files.
    csv_filename = os.path.join(
        base_dir, 'profiling_experiments', model_type, dataset,
        batching_method, 'round_' + str(batching_round_to_64), str(batch_size),
        'cpu',
        'iteration_0', 'graph_distribution_batching_path.csv')
    return csv_filename


def main(argv):
    # base_dir = 'tests/data'
    base_dir = '/home/dts/Documents/hu/batch_stats/batch_stats_data_dec_18_2024/u/dansp/batch_stats/batching_stats_mpeu_qm9_aflow'


    # create_histogram(
    #         base_dir, model_type='MPEU',
    #         batch_stats_col='num_node_before_batching', dataset='qm9',
    #         x_lims=(500, 620),
    #         y_lims=(0, 0.05))

    # Set fontsize to 13
    # create_nodes_and_edges_histogram(base_dir, model_type='MPEU', dataset='qm9')


    create_histogram_num_graphs_dynamic(base_dir, model_type='MPEU',
            batch_stats_col='num_graphs_before_batching', dataset='AFLOW')

if __name__ == '__main__':
    app.run(main)

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
BATCH_METHOD_LIST = ['static', 'dynamic']
N_BINS = 100
DATASET_LIST = ['AFLOW', 'qm9']

# import matplotlib.font_manager as font_manager
# font = font_manager.FontProperties(family=FONT,
#                                 # weight='bold',
#                                 style='normal', size=12)
FONTSIZE = 12
# FONT = 'Times'
# FONT = 'Times new roman'
FONT = 'serif'
ticksize=12
# def create_histogram(
#         base_dir, model_type='MPEU', batch_stats_col='num_node_before_batching', dataset='aflow'):
#     batching_round_to_64 = False
#     # for batching_method in BATCH_METHOD_LIST:
#         # set the subplot axis to bottom if dynamic.
#         # if batching_method == 'dynamic':
#         #     ax_num = 1
#         # else:
#         #     ax_num = 0
#     batching_method = 'static'
#     fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
#     # stats_array = get_batch_stats_data(base_dir, batch_stats_col, batching_method, dataset,
#     #                         batching_round_to_64=batching_round_to_64, model_type=model_type)
#     # Ok now let's loop over the dataset
#     for batching_method in BATCH_METHOD_LIST:
#         for col_index, dataset in enumerate(DATASET_LIST):
#             for row_index, batch_size in enumerate(BATCH_SIZE_LIST):
#                 try:
#                     stats_array = get_stats_for_single_batch_size(base_dir,
#                         batch_stats_col, batching_method, dataset, batching_round_to_64,
#                         batch_size, model_type)
#                     ax[row_index, col_index].hist(
#                         stats_array, N_BINS, density=True, histtype='bar')
#                         # color=COLORS_LIST, label=COLORS_LIST)
                    
#                 except FileNotFoundError:
#                     pass

#     # ax[ax_num].hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
#     # colors = ['red', 'tan', 'lime', 'black']
#     # ax.hist(
#     #     stats_array, N_BINS, density=True, histtype='bar', color=COLORS_LIST,
#     #     label=COLORS_LIST)

    
#     plt.show()



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
    # ax[1].text(0.1, 0.1, 'dynamic', fontsize=32)   
    # batch size 32, qm9
    # I think I should use step = 1!!!!
    bins=np.arange(x_lims[0], x_lims[1], step=0.5)+0.5 
    # batch size 64, qm9
    # bins=np.arange(1050,1200)+0.5
    # batch size 32, aflow
    # bins=np.arange(100,400)+0.5 


    # bins = 100   

    # ax[0].text(0.1, 0.1, 'static', fontsize=32)                

    stats_array = get_stats_for_single_batch_size(base_dir,
        batch_stats_col, 'dynamic', dataset, batching_round_to_64,
        batch_size, model_type)
    ax.hist(
        stats_array, bins, density=True, histtype='bar')
    ax.set_xlabel('Number of graphs in batch before padding', fontsize=12, font=FONT)
    ax.set_ylabel('Density', fontsize=12, font=FONT)

    ax.set_xlim(x_lims)


    # ax.text(0.8, 0.8, 'dynamic', horizontalalignment='center',
    #     verticalalignment='center', transform=ax.transAxes, fontsize=12)
    # ax[1].set_xlim([8000, 12000])

    # ax[0].set_ylim([0, 0.002])               
    # ax[1].set_ylim([0, 0.002])       

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
    # batch_size = 32
    # dataset = 'qm9'
    # model_type = 'MPEU'
    # fig, ax = plt.subplots(2, 1, figsize=(6.472, 4), sharex=True, sharey=True)
    fig, ax = plt.subplots(2, 1, figsize=(5.1, 5.1), sharex=True, sharey=True)


    # ax[1].text(0.1, 0.1, 'dynamic', fontsize=32)   
    # batch size 32, qm9
    # bins=np.arange(0,32) 
    # batch size 64, qm9
    bins=np.arange(x_lims[0],x_lims[1])+0.5
    # batch size 32, aflow
    # bins=np.arange(100,400)+0.5 


    # bins = 100



    stats_array = get_stats_for_single_batch_size(base_dir,
        batch_stats_col, 'static', dataset, batching_round_to_64,
        batch_size, model_type)
    ax[0].hist(
        stats_array, bins, density=True, histtype='bar')
    # ax[1].text(0.8, 0.8, 'static', horizontalalignment='center',
    #     verticalalignment='center', transform=ax[0].transAxes, fontsize=12)    

    # ax[0].text(0.1, 0.1, 'static', fontsize=32)                

    stats_array = get_stats_for_single_batch_size(base_dir,
        batch_stats_col, 'dynamic', dataset, batching_round_to_64,
        batch_size, model_type)
    ax[1].hist(
        stats_array, bins, density=True, histtype='bar')
    ax[1].set_xlabel(f'Number of {batch_stats_col.split("_")[1]}s in batch before padding',
                     fontsize=12, font=FONT)
    ax[0].set_ylabel('Density', fontsize=12, font=FONT)
    ax[1].set_ylabel('Density', fontsize=12, font=FONT)

    # stats_array = get_stats_for_single_batch_size(base_dir,
    #     'num_node_after_batching', 'dynamic', dataset, batching_round_to_64,
    #     batch_size, model_type)
    # print(stats_array[0:10])
    # ax[1].hist(
    #     stats_array, bins, density=True, histtype='bar')
    # ax[1].text(0.8, 0.8, 'dynamic', horizontalalignment='center',
    #     verticalalignment='center', transform=ax[1].transAxes, fontsize=12)
    # ax[1].set_xlim([500, 620]) bs 32, qm9

    # ax[0].set_ylim([0, 0.05])               
    # ax[1].set_ylim([0, 0.05])  

    # ax[1].set_xlim([8000, 12000])
    ax[1].set_xlim([x_lims[0], x_lims[1]])
    # ax[0].set_ylim([0, 0.002])               
    # ax[1].set_ylim([0, 0.002])       
    ax[0].set_ylim([y_lims[0], y_lims[1]])               
    ax[1].set_ylim([y_lims[0], y_lims[1]])     

    plt.tight_layout()
    plt.savefig(f'/home/dts/Documents/theory/batching_paper/figs/batch_stats_{batch_stats_col}_size_{str(batch_size)}_{dataset}.png', dpi=600)
    plt.show()


def create_nodes_and_edges_histogram(base_dir, model_type='MPEU',
        dataset='aflow',
        x1_lims=(500,620),
        x2_lims=(8000,12000),
        y1_lims=(0, 0.05),
        y2_lims=(0, 0.002),
        batch_size=32):
    batching_round_to_64 = False


    # Column names to pull
    num_edges_col= 'num_edge_before_batching'
    num_nodes_col= 'num_node_before_batching'

    fig, ax = plt.subplots(2, 1, figsize=(5.1, 5.1), sharex=False, sharey=False)

    bins_lhs = np.arange(x1_lims[0], x1_lims[1])+0.5

    bins_rhs = 120 # np.arange(x2_lims[0], x2_lims[1])+0.5

    node_stats_array_static = get_stats_for_single_batch_size(base_dir,
        num_nodes_col, 'static', dataset, batching_round_to_64,
        batch_size, model_type)
    node_stats_array_dynamic = get_stats_for_single_batch_size(base_dir,
        num_nodes_col, 'dynamic', dataset, batching_round_to_64,
        batch_size, model_type)
    ax[0].hist(
        node_stats_array_static, bins_lhs, density=True, histtype='bar', alpha=0.5, label='static')
    ax[0].hist(
        node_stats_array_dynamic, bins_lhs, density=True, histtype='bar', alpha=0.5, label='dynamic')

    edge_stats_array_static = get_stats_for_single_batch_size(base_dir,
        num_edges_col, 'static', dataset, batching_round_to_64,
        batch_size, model_type)
    edge_stats_array_dynamic = get_stats_for_single_batch_size(base_dir,
        num_edges_col, 'dynamic', dataset, batching_round_to_64,
        batch_size, model_type)


    ax[1].hist(
        edge_stats_array_static, bins_rhs, density=True, histtype='bar', alpha=0.5, label='static')
    ax[1].hist(
        edge_stats_array_dynamic, bins_rhs, density=True, histtype='bar', alpha=0.5, label='dynamic')
    
    ax[0].set_xlabel(f'Nodes in batch before padding',
                     fontsize=12, font=FONT)
    ax[1].set_xlabel(f'Edges in batch before padding',
                     fontsize=12, font=FONT)
    ax[0].set_ylabel('Density', fontsize=12, font=FONT)
    ax[1].set_ylabel('Density', fontsize=12, font=FONT)

    # ax[0].tick_params(which='both', labelsize=ticksize)
    # ax[1].tick_params(which='both', labelsize=ticksize)

    ax[0].set_xlim([x1_lims[0], x1_lims[1]])

    ax[1].set_xlim([x2_lims[0], x2_lims[1]])
    
    ax[0].set_ylim([y1_lims[0], y1_lims[1]])                    
    ax[1].set_ylim([y2_lims[0], y2_lims[1]])                    
    ax[1].set_xticks([8000, 9000, 10000, 11000, 12000])
    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=12)
    # ax.legend(prop=font)

    ax[1].legend(loc='upper left', prop=font, edgecolor="black", fancybox=False)
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

    create_nodes_and_edges_histogram(base_dir, model_type='MPEU', dataset='qm9')


    # create_histogram_num_graphs_dynamic(base_dir, model_type='MPEU',
    #         batch_stats_col='num_graphs_before_batching', dataset='AFLOW')

if __name__ == '__main__':
    app.run(main)

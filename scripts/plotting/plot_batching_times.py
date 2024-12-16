""" Plot batching times.

Let's make one plot for the batching times for all options:

static
round_True
jnp
np

dynamic

And one plot for the update functions.

First plot, will be for GPU only and SchNett.

Second plot will be for CPU only and SchNett.

Third plot with GPU only and MPEU.

Fourth plot will be CPU only and MPEU.


Let's use the dataframes directly.
"""
from absl import app
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


BASE_DIR = '/home/dts/Documents/hu'
# STATIC_NP_PROFILING_CSV = 'parsed_profiling_static_batching_seb_fix_qm9_aflow_schnet_mpeu_100k_steps_23_05__27_06_2024.csv'
# PROFILING_JNP_CSV = 'parsed_profiling_qm9_aflow_schnet_100k_steps_23_54__8_03_2024.csv'
# ROUND_PROFILING_JNP_CSV = 'parsed_profiling_round_to_multiple_qm9_aflow_schnet_100k_steps_23_54__8_03_2024.csv'

COMBINED_CSV = 'parsed_profiling_static_batching_seb_fix_qm9_aflow_schnet_mpeu_100k_steps_16_05__12_12_2024.csv'

BATCH_SIZE_DICT = {
    '16': 0,
    '32': 1,
    '64': 2,
    '128': 3,
}

BATCH_SIZE_LIST = [16, 32, 64, 128]

# MODEL_TYPE_LIST = ['schnet', 'mpeu']
MODEL_TYPE_LIST = ['schnet', 'MPEU']

# BATCH_METHOD_LIST = ['dynamic', 'static']
BATCH_METHOD_LIST = ['dynamic', 'static']
# COMPUTING_TYPE_LIST = ['gpu_a100', 'cpu']
COMPUTING_TYPE_LIST = ['gpu_a100']

DATASET_LIST = ['aflow']


def plot_four_plots(df):
    """Plot all four 4x4 subplot plots.
    
    Each plot is a different ML model + hardware platform

    AFLOW in the left columns, qm9 in the right columns.
    Batching time in the top rows and update time in the bottom rows.

    This script runs through a for loop.
    """

    for model in MODEL_TYPE_LIST:
        for compute_type in COMPUTING_TYPE_LIST:
            plot_batching_update_subplot(df, model, compute_type)
            pass


def get_avg_std_of_profile_column(
        df, profile_column, model, batch_method, compute_type, batch_size, dataset, batching_round_to_64):

    # Now keep only the rows we want.
    # df = df[
    #     (df['model'] == model) & (df['batching_type'] == batch_method)
    #     & (df['computing_type'] == compute_type) & (df['batch_size'] == batch_size)
    #     & (df['dataset'] == dataset)]
    # Now get the standard devation across different rows.
    # return df.mean(skipna = True), df.std(skipna = True)

    profile_column_dict = {
        'batching': 'step_100000_batching_time_mean',
        'update': 'step_100000_update_time_mean'
    }

    gpu_profiling_df = df[
        (df['dataset'] == dataset) & (df['model'] == model) &
        (df['batching_type'] == batch_method) &
        (df['computing_type'] == compute_type) &
        (df['batch_size'] == batch_size) &
        (df['batching_round_to_64'] == batching_round_to_64)]
    # gpu_profiling_df = gpu_profiling_df[gpu_profiling_df['computing_type'] == compute_type]
    # gpu_profiling_df = gpu_profiling_df[gpu_profiling_df['model'] == model]
    # gpu_profiling_df = gpu_profiling_df[gpu_profiling_df['batching_round_to_64'] == batching_round_to_64]
    # gpu_profiling_df = gpu_profiling_df[gpu_profiling_df['batch_size'] == batch_size]
    gpu_profiling_df = gpu_profiling_df[profile_column_dict[profile_column]]


    # TODO, fix this so that we are using the averages over the ten iterations.
    # gpu_profiling_df = gpu_profiling_df.rename(columns={
    #     "batch_size": "Batch size", "batching_type": "Algorithm",
    #     'step_100000_batching_time_mean': 'Mean batching time (s)',
    #     'step_100000_update_time_mean': 'Mean update time (s)'})

    # gpu_profiling_df_grouped = gpu_profiling_df.groupby(['Algorithm', 'Batch size'])
    # list_of_cols_to_display = ['Batch size', 'Algorithm', 'Mean batching time (s)', 'Mean update time (s)']

    # pd.options.display.float_format = "{:,.5f}".format

    # gpu_profiling_df_grouped = gpu_profiling_df_grouped[list_of_cols_to_display]

    # gpu_profiling_df_grouped.mean()

    # print(f'raw data before average: {gpu_profiling_df}')
    mean_result, std_result = gpu_profiling_df.mean(), gpu_profiling_df.std()

    return mean_result, std_result


def plot_batching_update_subplot(df, model, compute_type):
    """Here we want to feed into the correct dataframes to be able to easily make a four panel plot
    
    AFLOW in the left colum, qm9 in the right column.
    Batching time in the top row, update time in the bottom row.
    """
    batching_aflow_list = []
    update_aflow_list = []

    batching_aflow_list = []
    update_aflow_list = []

    aflow_batch_axes = [0, 0]
    fig, ax = plt.subplots(2, 2, figsize=(7, 7)) #, gridspec_kw={'height_ratios': [1]})

    aflow_batching_axes = ax[0, 0]
    qm9_batching_axes = ax[0, 1]
    aflow_update_axes = ax[1, 0]
    qm9_update_axes = ax[1, 1]

    axes_list = [aflow_batching_axes, qm9_batching_axes, aflow_update_axes, qm9_update_axes]
    plot_content_list = [('aflow', 'batching'), ('qm9', 'batching'), ('aflow', 'update'),
                         ('qm9', 'update')]
    # color_list = ['k', 'r', 'b', 'y']
    color_list = ['k', 'r']
    marker_list = ['x', '^']



    for plot_num in range(len(axes_list)):
        dataset = plot_content_list[plot_num][0]  # Should be either AFLOW or qm9
        profile_column = plot_content_list[plot_num][1]  # Either `batching` or `update`
        batching_round_to_64 = True
        for color_counter, batch_method in enumerate(BATCH_METHOD_LIST):
            y_mean_list = []
            y_std_list = []
            for batch_size in BATCH_SIZE_LIST:
                y_mean, y_std = get_avg_std_of_profile_column(
                    df, profile_column, model, batch_method, compute_type,
                    batch_size, dataset, batching_round_to_64)
                y_mean_list.append(y_mean)
                y_std_list.append(y_std)
            print(f' the batch method is: {batch_method}')
            print(f' the y mean list is {y_mean_list}')
            print(f' the profile col is: {profile_column}')
            print(f' the dataset is: {dataset}')

            axes_list[plot_num].plot(BATCH_SIZE_LIST, np.multiply(y_mean_list, 1000),
                                     marker_list[color_counter],
                                     markersize=11, alpha=1,
                                     color=color_list[color_counter])
    
    # ax[0, 0].set_xlim(0, 5)
    # ax[0, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
    ax[0, 0].set_ylabel('Time (ms)', fontsize=12)
    # ax[0, 0].set_yscale('log')
    # ax[0, 0].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)
    ax[0, 0].set_ylim(0, 3)


    # ax[0, 1].set_xlim(0, 5)
    # ax[0, 1].set_ylim(0, 10)
    # ax[0, 1].set_yscale('log')
    # ax[0, 1].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)

    # ax[0, 1].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
    ax[0, 1].set_ylim(0, 3)

    # ax[1, 0].set_xlim(0, 5)
    # ax[1, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
    ax[1, 0].set_xlabel('Batch size', fontsize=12)
    ax[1, 0].set_ylabel('Time (ms)', fontsize=12)
    # ax[1, 0].set_yscale('log')
    # ax[1, 0].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)
    ax[1, 0].set_ylim(0, 10)

    # ax[1, 1].set_xlim(0, 5)
    # ax[1, 1].set_ylim(0, 100)
    # ax[1, 1].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
    ax[1, 1].set_xlabel('Batch size', fontsize=12)
    # ax[1, 1].set_yscale('log')
    # ax[1, 1].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)

    ax[1, 1].set_ylim(0, 10)

    plt.show()

def main(argv):
    # plot learning curves
    df = pd.read_csv(os.path.join(BASE_DIR, COMBINED_CSV))
    # Ok now let's plot the batching times. Let's plot 4 graphs.
    # AFLOW / SchNet (GPU / CPU)
    plot_batching_update_subplot(df, model='MPEU', compute_type='gpu_a100')


if __name__ == '__main__':
    app.run(main)

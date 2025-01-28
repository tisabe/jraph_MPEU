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
import sys
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from matplotlib import rc, font_manager

# import matplotlib as mpl
# label_size = 12
# mpl.rcParams['xtick.labelsize'] = label_size 


BASE_DIR = '/home/dts/Documents/hu/jraph_MPEU/batch_data'
# STATIC_NP_PROFILING_CSV = 'parsed_profiling_static_batching_seb_fix_qm9_aflow_schnet_mpeu_100k_steps_23_05__27_06_2024.csv'
# PROFILING_JNP_CSV = 'parsed_profiling_qm9_aflow_schnet_100k_steps_23_54__8_03_2024.csv'
# ROUND_PROFILING_JNP_CSV = 'parsed_profiling_round_to_multiple_qm9_aflow_schnet_100k_steps_23_54__8_03_2024.csv'

# COMBINED_CSV = 'parsed_profiling_static_batching_seb_fix_qm9_aflow_schnet_mpeu_100k_steps_16_05__12_12_2024.csv'

COMBINED_CSV = 'parsed_profiling_static_batching_seb_fix_qm9_aflow_schnet_mpeu_100k_steps_15_12__30_12_2024.csv'
# COMBINED_CSV = 'parsed_profiling_static_batching_seb_fix_qm9_aflow_schnet_mpeu_100k_steps_11_31__23_12_2024.csv'
# COMBINED_CSV = 'parsed_profiling_batching_2_000_000_steps_aflow_qm9_20_12_2024.csv'

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
BATCH_METHOD_LIST = ['dynamic', 'static', 'static-64']
# BATCH_METHOD_LIST = ['static-2']

# COMPUTING_TYPE_LIST = ['gpu_a100', 'cpu']
COMPUTING_TYPE_LIST = ['gpu_a100']

DATASET_LIST = ['aflow']

FONTSIZE = 12
FONT = 'serif'

fontProperties = {'family':'sans-serif','sans-serif':['Times'],
    'weight' : 'normal', 'size' : FONTSIZE}
ticks_font = font_manager.FontProperties(family='Times', style='normal',
    size=FONTSIZE, weight='normal', stretch='normal')
# rc('text', usetex=True)
rc('text')
rc('font',**fontProperties)

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
        df, profile_column, model, batch_method, compute_type, batch_size,
        dataset, batching_round_to_64, mean_or_median='mean'):


    profile_column_dict = {
        'batching': f'step_100000_batching_time_{mean_or_median}',
        'update': f'step_100000_update_time_{mean_or_median}',
        'recompilation': 'recompilation_counter',
        'combined': f'step_100000_update_time_{mean_or_median}'
    }


    gpu_profiling_df = df[
        (df['dataset'] == dataset) & (df['model'] == model) &
        (df['batching_type'] == batch_method) &
        (df['computing_type'] == compute_type) &
        (df['batch_size'] == batch_size) &
        (df['batching_round_to_64'] == batching_round_to_64)]

    gpu_profiling_df_col = gpu_profiling_df[profile_column_dict[profile_column]]


    if mean_or_median == 'mean':
        mean_result, std_result = gpu_profiling_df_col.mean(), gpu_profiling_df_col.std()
        if profile_column == 'update':
            ## Then we need to subtract the mean batching time and add the batching time standard dev.
            gpu_profiling_df_batching = gpu_profiling_df[profile_column_dict['batching']]
            mean_batching, std_batching = gpu_profiling_df_batching.mean(), gpu_profiling_df_batching.std()
            mean_result = mean_result - mean_batching
            std_result = std_result + std_batching            
    elif mean_or_median == 'median':
        mean_result, std_result = gpu_profiling_df_col.median(), gpu_profiling_df_col.std()
        if profile_column == 'update':
            ## Then we need to subtract the mean batching time and add the batching time standard dev.
            gpu_profiling_df_batching = gpu_profiling_df[profile_column_dict['batching']]
            median_batching, std_batching = gpu_profiling_df_batching.mean(), gpu_profiling_df_batching.std()
            mean_result = mean_result - median_batching
            std_result = std_result + std_batching     
    else:
        raise(ValueError)
    return mean_result, std_result


def plot_batching_update_subplot(df, model, compute_type, mean_or_median):
    """Here we want to feed into the correct dataframes to be able to easily make a four panel plot
    
    AFLOW in the left colum, qm9 in the right column.
    Batching time in the top row, update time in the bottom row.
    """
    batching_aflow_list = []
    update_aflow_list = []

    batching_aflow_list = []
    update_aflow_list = []

    aflow_batch_axes = [0, 0]
    # previously was 7,7
    fig, ax = plt.subplots(3, 2, figsize=(5.1, 7.65)) #, gridspec_kw={'height_ratios': [1]})

    aflow_batching_axes = ax[0, 0]
    qm9_batching_axes = ax[0, 1]
    aflow_update_axes = ax[1, 0]
    qm9_update_axes = ax[1, 1]
    aflow_combined_axes = ax[2, 0]
    qm9_combined_axes = ax[2, 1]

    axes_list = [
        aflow_batching_axes, qm9_batching_axes, aflow_update_axes, qm9_update_axes,
        aflow_combined_axes, qm9_combined_axes]
    # plot_content_list = [('aflow', 'batching'), ('qm9', 'batching'), ('aflow', 'update'),
    #                      ('qm9', 'update')]
    plot_content_list = [('aflow', 'batching'), ('qm9', 'batching'), ('aflow', 'update'),
                         ('qm9', 'update'), ('aflow', 'combined'), ('qm9', 'combined')]
    # color_list = ['k', 'r', 'b', 'y']
    color_list = ['#1f77b4', '#ff7f0e', '#9467bd']
    marker_list = ['x', '^', '.']


    for plot_num in range(len(axes_list)):
        dataset = plot_content_list[plot_num][0]  # Should be either AFLOW or qm9
        profile_column = plot_content_list[plot_num][1]  # Either `batching` or `update`
        batching_round_to_64 = False
        for color_counter, batch_method in enumerate(BATCH_METHOD_LIST):
            y_mean_list = []
            y_std_list = []
            label = batch_method
            if batch_method == 'static-64':
                batching_round_to_64 = True
                batch_method = 'static'
                label = 'static-$64$'
            elif batch_method == 'static':
                batching_round_to_64 = False
                batch_method = 'static'
                label = 'static-$2^N$'

            # elif batch_method == 'static':
            #     batching_round_to_64 = True
            else:
                batching_round_to_64 = True
                # sys.err(f'error wrong batch method {batch_method}')
            for batch_size in BATCH_SIZE_LIST:

                print(f'batch rond to 64 is set to {batching_round_to_64}')

                y_mean, y_std = get_avg_std_of_profile_column(
                    df, profile_column, model, batch_method, compute_type,
                    batch_size, dataset, batching_round_to_64, mean_or_median)
                y_mean_list.append(y_mean)
                y_std_list.append(y_std)
            

            print(f' the batch method is: {batch_method}')
            print(f' the y mean list is {y_mean_list}')
            # print(f' the profile col is: {profile_column}')
            print(f' the dataset is: {dataset}')
            print(f'The std is {y_std}\n')

            axes_list[plot_num].errorbar(BATCH_SIZE_LIST,
                                     np.multiply(y_mean_list, 1000), yerr=y_std,
                                     marker=marker_list[color_counter],
                                     markersize=11, alpha=0.9,
                                     color=color_list[color_counter],
                                     label=label, linestyle='')
    xlim = [0, 140]
    ax[0, 0].set_ylabel('Batching time (ms)', fontsize=FONTSIZE, font=FONT)
    ax[0, 0].set_ylim(0, 8)
    ylabels = [0, 2, 4, 6, 8]
    ax[0, 0].set_yticks(ylabels, font=FONT, fontsize=FONTSIZE)

    ax[0, 0].set_yticklabels(ylabels, font=FONT, fontsize=FONTSIZE, rotation=0)
    ax[0, 1].set_ylim(0, 8)
    ax[0, 1].set_yticklabels([])

    ax[0, 0].set_xticks([16, 32, 64, 128])
    ax[0, 0].set_xticklabels([])


    ax[0, 0].set_title('AFLOW', font=FONT, fontsize=FONTSIZE)
    ax[0, 1].set_title('QM9', font=FONT, fontsize=FONTSIZE)

    ax[0, 1].set_xticks([16, 32, 64, 128])
    ax[0, 1].set_xticklabels([])
    ax[0, 0].set_xlim(xlim[0], xlim[1])
    ax[0, 1].set_xlim(xlim[0], xlim[1]) 
    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)

    ax[0, 0].legend(loc='upper left', prop=font, edgecolor="black", fancybox=False)

    ax[1, 0].set_xticklabels([])
    ax[1, 0].set_ylabel('Update time (ms)', fontsize=FONTSIZE, font=FONT)


    if compute_type == 'cpu':
        ylim = 200
        ylabels = [0, 50, 100, 150, 200]
        ax[0, 1].text(12, 6.5, 'CPU only', font=FONT, fontsize=FONTSIZE)
    else:
        ax[0, 1].text(12, 6.5, 'GPU+CPU', font=FONT, fontsize=FONTSIZE)
        ylim = 15
        ylabels = [0, 5, 10, 15]

    ax[0, 1].text(12, 4.5, mean_or_median, font=FONT, fontsize=FONTSIZE)
    ax[0, 1].text(12, 3.5, '100,000 steps', font=FONT, fontsize=FONTSIZE)

    if model == 'schnet':
        model_label = 'SchNet'
    else:
        model_label = model
    ax[0, 1].text(12, 5.5, model_label, font=FONT, fontsize=FONTSIZE)

    ax[1, 0].set_yticks(ylabels, font=FONT, fontsize=FONTSIZE)
    ax[1, 0].set_yticklabels(ylabels, font=FONT, fontsize=FONTSIZE, rotation=0)
    ax[1, 0].set_ylim(0, ylim)

    ax[1, 1].set_xticklabels([])

    ax[1, 1].set_ylim(0, ylim)
    ax[1, 1].set_yticklabels([])
    ax[1, 0].set_xticks([16, 32, 64, 128])
    ax[1, 1].set_xticks([16, 32, 64, 128])
    ax[1, 0].set_xlim(xlim[0], xlim[1])
    ax[1, 1].set_xlim(xlim[0], xlim[1]) 
    ax[2, 0].set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)
    ax[2, 0].set_ylabel('Combined time (ms)', fontsize=FONTSIZE, font=FONT)

    ax[2, 0].set_ylim(0, ylim)
    # ylabels = [0, 50, 100, 150, 200]
    ax[2, 0].set_yticks(ylabels, font=FONT, fontsize=FONTSIZE)
    ax[2, 0].set_yticklabels(ylabels, font=FONT, fontsize=FONTSIZE, rotation=0)

    ax[2, 1].set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)

    ax[2, 1].set_ylim(0, ylim)
    ax[2, 1].set_yticklabels([])

    ax[2, 0].set_xticks([16, 32, 64, 128], fontsize=FONTSIZE)
    ax[2, 1].set_xticks([16, 32, 64, 128], fontsize=FONTSIZE)
    ax[2, 0].set_xticklabels([16, 32, 64, 128], font=FONT, fontsize=FONTSIZE, rotation=0)
    ax[2, 1].set_xticklabels([16, 32, 64, 128], font=FONT, fontsize=FONTSIZE, rotation=0)
    ax[2, 0].set_xlim(xlim[0], xlim[1])
    ax[2, 1].set_xlim(xlim[0], xlim[1]) 


    # for i in range(2):
    #     for j in range(2):
    #         ax[i, j].tick_params(axis='both', which='minor', labelsize=12)
    #         ax[i, j].tick_params(axis='both', which='major', labelsize=12)
    #         plt.setp(ax[i,j].get_xticklabels(), fontsize=12, font=FONT) 
    #                 # horizontalalignment="left")
    #         plt.setp(ax[i,j].get_yticklabels(), fontsize=12, font=FONT)
    #                 # horizontalalignment="left")

    plt.style.use(["science", "grid"])
    fig.align_labels()

    plt.tight_layout()
    plt.savefig(
        f'/home/dts/Documents/theory/batching_paper/figs/profiling_100k_{model}_{compute_type}_aflow_left_qm9_right_{mean_or_median}.png',
        dpi=600)
    plt.show()

def plot_recompilation_bar_plot(df):
    """Create bar plot of # of recompilations.
    
    X-axis is the batch size.
    Y-axis is the number of recompilations.
    """

    # Keep the batch size, batching method, round true, and recompilation number
    # df = df[['batch_size', 'batching_round_to_64', 'dataset', 'computing_type', 'batching_type', 'recompilation_counter']]
    profile_column = 'recompilation'
    computing_type = 'gpu_a100'
    model = 'MPEU'
    dataset = 'aflow'
    # batch_method = ''

    # recompilation_list = []
    # for batch_size in BATCH_SIZE_LIST:
    #     for batch_method in BATCH_METHOD_LIST:

    # avg_recompilation, recompilation_std = get_avg_std_of_profile_column(
    #     df, profile_column, model, batch_method, compute_type,
    #     batch_size, dataset, batching_round_to_64)

    # Create a new batching method, batch-64 based on rounding.
    df.loc[(df.batching_type == 'static') & (df.batching_round_to_64 == True),'batching_type'] ='static-64'
    # Get data only for gpu and AFLOW and MPEU
    df = df[df['model'] == model]
    df = df[df['computing_type'] == computing_type]
    df = df[df['dataset'] == dataset]

    # Now take the mean over the different iterations.
    df = df[['batch_size', 'batching_type', 'recompilation_counter']]
    df = df.groupby(['batch_size', 'batching_type']).mean()
    print(df)
    # fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # Let's now remove 
    # previously size 5.1, 4
    ax = df.unstack().plot.bar(figsize=(5.1, 5.1))

    # h,l = ax.get_legend_handles_labels()
    # ax.legend(h[:3],["dynamic", "static", "static-64"], loc=3, fontsize=12)

    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=12)

    # ax.legend(["dynamic", "static", "static-64"]);
    plt.legend(
        ["dynamic", "static", "static-64"], fontsize=12,
        prop=font, edgecolor="black", fancybox=False)
    ax.set_xlabel('Batch size', fontsize=12, font=FONT)
    ax.set_ylabel('Number of recompilations', fontsize=12, font=FONT)


    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.setp(ax.get_xticklabels(), fontsize=12, font=FONT) 
            # horizontalalignment="left")
    plt.setp(ax.get_yticklabels(), fontsize=12, font=FONT)
    plt.tight_layout()
    plt.savefig(
        '/home/dts/Documents/theory/batching_paper/figs/recompilation_count_data_{dataset}.png',
        dpi=600)
    plt.show()


def main(argv):
    # plot learning curves
    df = pd.read_csv(os.path.join(BASE_DIR, COMBINED_CSV))
    # Ok now let's plot the batching times. Let's plot 4 graphs.
    # AFLOW / SchNet (GPU / CPU)
    # plot_batching_update_subplot(df, model='MPEU',
    #                              compute_type='cpu',
    #                              mean_or_median='median')
                                #  compute_type='gpu_a100')
    plot_recompilation_bar_plot(df)

if __name__ == '__main__':
    app.run(main)

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
## The 2 million profiling steps data:
COMBINED_CSV = 'parsed_profiling_batching_2_000_000_steps_combined_19_01_2025.csv'
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
BATCH_METHOD_LIST = ['dynamic', 'static-64', 'static']
# BATCH_METHOD_LIST = ['static-2']

# COMPUTING_TYPE_LIST = ['gpu_a100', 'cpu']
COMPUTING_TYPE_LIST = ['gpu_a100']

DATASET_LIST = ['aflow']

FONTSIZE = 12
# FONT = 'Times'
# FONT = 'Times new roman'
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

    # Now keep only the rows we want.
    # df = df[
    #     (df['model'] == model) & (df['batching_type'] == batch_method)
    #     & (df['computing_type'] == compute_type) & (df['batch_size'] == batch_size)
    #     & (df['dataset'] == dataset)]
    # Now get the standard devation across different rows.
    # return df.mean(skipna = True), df.std(skipna = True)

    profile_column_dict = {
        'batching': f'step_2_000_000_batching_time_{mean_or_median}',
        'update': f'step_2_000_000_update_time_{mean_or_median}',
        'recompilation': 'recompilation_counter',
    }

    # profile_column_dict = {
    #     'batching': 'step_2000000_batching_time_mean',
    #     'update': 'step_2000000_update_time_mean'
    # }

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
    if mean_or_median == 'mean':
        mean_result, std_result = gpu_profiling_df.mean(), gpu_profiling_df.std()
    elif mean_or_median == 'median':
        mean_result, std_result = gpu_profiling_df.median(), gpu_profiling_df.std()
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
    fig, ax = plt.subplots(2, 2, figsize=(5.1, 5.1)) #, gridspec_kw={'height_ratios': [1]})

    aflow_batching_axes = ax[0, 0]
    qm9_batching_axes = ax[0, 1]
    aflow_update_axes = ax[1, 0]
    qm9_update_axes = ax[1, 1]

    axes_list = [aflow_batching_axes, qm9_batching_axes, aflow_update_axes, qm9_update_axes]
    plot_content_list = [('aflow', 'batching'), ('qm9', 'batching'), ('aflow', 'update'),
                         ('qm9', 'update')]
    # color_list = ['k', 'r', 'b', 'y']
    color_list = ['k', 'r', 'b']
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
            elif batch_method == 'static':
                batching_round_to_64 = False
                batch_method = 'static'
            # elif batch_method == 'static':
            #     batching_round_to_64 = True
            else:
                batching_round_to_64 = False  # Different than the 200k experiments.
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

            axes_list[plot_num].plot(BATCH_SIZE_LIST,
                                     np.multiply(y_mean_list, 1000),
                                     marker_list[color_counter],
                                     markersize=11, alpha=0.8,
                                     color=color_list[color_counter],
                                     label=label)
    
    # ax[0, 0].set_xlim(0, 5)
    # ax[0, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
    ax[0, 0].set_ylabel('Batching time (ms)', fontsize=FONTSIZE, font=FONT)
    # ax[0, 0].set_yscale('log')
    # ax[0, 0].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)
    ax[0, 0].set_ylim(0, 5)


    # ax[0, 1].set_xlim(0, 5)
    # ax[0, 1].set_ylim(0, 10)
    # ax[0, 1].set_yscale('log')
    # ax[0, 1].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)

    # ax[0, 1].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
    ax[0, 1].set_ylim(0, 5)
    ax[0, 1].set_yticklabels([])

    # ax[1, 0].set_xlim(0, 5)
    # ax[1, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
    ax[1, 0].set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)
    ax[1, 0].set_ylabel('Update time (ms)', fontsize=FONTSIZE, font=FONT)
    # ax[1, 0].set_yscale('log')
    # ax[1, 0].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)
    ax[1, 0].set_ylim(0, 18)
    # ax[1, 0].set_ylim(0, 200)

    ax[1, 1].set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)


    ax[1, 1].set_ylim(0, 18)
    ax[1, 1].set_yticklabels([])

    # ax[1, 1].set_ylim(0, 200)

    # ax[1, 0].text(0.25, 0.9, f'update time', horizontalalignment='center',
    #     verticalalignment='center', transform=ax[1, 0].transAxes, fontsize=12)
    # ax[1, 1].text(0.25, 0.9, f'update time', horizontalalignment='center',
    #     verticalalignment='center', transform=ax[1, 1].transAxes, fontsize=12)
    # ax[0, 1].text(0.25, 0.9, f'batching time', horizontalalignment='center',
    #     verticalalignment='center', transform=ax[0, 1].transAxes, fontsize=12)
    # ax[0, 0].text(0.25, 0.9, f'batching time', horizontalalignment='center',
    #     verticalalignment='center', transform=ax[0, 0].transAxes, fontsize=12)
    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)
    # ax.legend(prop=font)

    ax[1, 0].legend(loc='upper left', prop=font, edgecolor="black", fancybox=False)
    # ax[1, 1].legend(loc='lower right')
    for i in range(2):
        for j in range(2):
            ax[i, j].tick_params(axis='both', which='minor', labelsize=FONTSIZE-2)
            ax[i, j].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
            plt.setp(ax[i,j].get_xticklabels(), fontsize=FONTSIZE-2, font=FONT) 
                    # horizontalalignment="left")
            plt.setp(ax[i,j].get_yticklabels(), fontsize=FONTSIZE-2, font=FONT)
                    # horizontalalignment="left")
        # for label in ax[i, j].get_xticklabels():
        #     label.set_fontproperties(ticks_font)

        # for label in ax[i, j].get_yticklabels():
        #     label.set_fontproperties(ticks_font)

    # plt.xticks(fontname = FONT, fontsize=12)  # This argument will change the font. 

    # plt.yticks(fontname = FONT, fontsize=12)  # This argument will change the font. 

    plt.style.use(["science", "grid"])
    fig.align_labels()

    plt.tight_layout()
    plt.savefig(
        f'/home/dts/Documents/theory/batching_paper/figs/profiling_2mill_{model}_gpu_aflow_left_qm9_right_{mean_or_median}.png',
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
    dataset = 'qm9'
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
    df = df[df['dataset'] == 'qm9']

    # Now take the mean over the different iterations.
    df = df[['batch_size', 'batching_type', 'recompilation_counter']]
    df = df.groupby(['batch_size', 'batching_type']).mean()

    # fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # Let's now remove 
    # previously size 5.1, 4
    ax = df.unstack().plot.bar(figsize=(5.1, 5.1))

    # h,l = ax.get_legend_handles_labels()
    # ax.legend(h[:3],["dynamic", "static", "static-64"], loc=3, fontsize=12)

    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)

    # ax.legend(["dynamic", "static", "static-64"]);
    plt.legend(
        ["dynamic", "static", "static-64"], fontsize=FONTSIZE,
        prop=font, edgecolor="black", fancybox=False)
    ax.set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)
    ax.set_ylabel('Number of recompilations', fontsize=FONTSIZE, font=FONT)


    ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE-2)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    plt.setp(ax.get_xticklabels(), fontsize=FONTSIZE-2, font=FONT) 
            # horizontalalignment="left")
    plt.setp(ax.get_yticklabels(), fontsize=FONTSIZE-2, font=FONT)
    plt.tight_layout()
    plt.savefig(
        '/home/dts/Documents/theory/batching_paper/figs/recompilation_count.png',
        dpi=600)
    plt.show()


def main(argv):
    # plot learning curves
    df = pd.read_csv(os.path.join(BASE_DIR, COMBINED_CSV))
    # Ok now let's plot the batching times. Let's plot 4 graphs.
    # AFLOW / SchNet (GPU / CPU)
    plot_batching_update_subplot(df, model='schnet',
                                 compute_type='gpu_a100',
                                 mean_or_median='mean')
                                #  compute_type='gpu_a100')
    # plot_recompilation_bar_plot(df)

if __name__ == '__main__':
    app.run(main)

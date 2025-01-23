import argparse
import os
import pickle
from absl import flags
from absl import app
import pandas as pd
import scienceplots

import matplotlib.pyplot as plt
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'csv_filename',
    'None',
    'Where to store data as csv that has been parsed.')


BASE_DIR = '/home/dts/Documents/hu/batch_stats'
# COMBINED_CSV = 'parsed_profiling_batching_2_000_000_steps_aflow_qm9_20_12_2024.csv'
DYNAMIC_CSV = 'time_series_dynamic_qm9_mpeu_gpu.csv'
STATIC_64_CSV = 'time_series_static_64_qm9_mpeu_gpu.csv'
STATIC_CSV = ''

FONTSIZE = 14
FONT = 'serif'



def plot_curves():
    color_list = ['k', 'r', 'b']
    marker_list = ['x', '^', '.']
    fig, ax = plt.subplots(1, 1, figsize=(5.1, 4))
    dynamic_df = pd.read_csv(os.path.join(BASE_DIR, DYNAMIC_CSV))
    stats_64_df = pd.read_csv(os.path.join(BASE_DIR, STATIC_64_CSV))

    ax.plot(
        dynamic_df['step_number']/1000000,
        dynamic_df['combined_time'],
        marker_list[0],
        alpha=0.4, markersize=8,
        label='dynamic', color=color_list[0])

    ax.plot(
        stats_64_df['step_number']/1000000,
        stats_64_df['combined_time'],
        marker_list[1],
        alpha=0.4, markersize=8,
        label='static-64', color=color_list[1])


    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)

    ax.legend(loc='lower right', prop=font, edgecolor="black", fancybox=False)

    ax.set_ylabel('Combined time (ms)', fontsize=FONTSIZE, font=FONT)
    ax.set_xlabel('Training steps (millions)', fontsize=FONTSIZE, font=FONT)
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 10])

    # axs[3, 1].set_xlabel('Steps (millions)', fontsize=12, font=FONT)
    # plt.style.use(["science", "grid"])

    plt.tight_layout()

    plt.savefig(
        f'/home/dts/Documents/theory/batching_paper/figs/recompilations_time_series_batch_size_32_qm9_MPEU.png',
        dpi=600)
    plt.show()

def plot_running_average_curves():
    color_list = ['k', 'r', 'b']
    marker_list = ['x', '^', '.']
    fig, ax = plt.subplots(1, 1, figsize=(5.1, 4))
    dynamic_df = pd.read_csv(os.path.join(BASE_DIR, DYNAMIC_CSV))
    static_64_df = pd.read_csv(os.path.join(BASE_DIR, STATIC_64_CSV))

    time_steps = list(np.linspace(0, 2000000, 21))
    running_avg_dynamic = running_average(dynamic_df['combined_time'].to_list(), time_steps)
    running_avg_static_64 = running_average(static_64_df['combined_time'].to_list(), time_steps)



    ax.plot(np.array(time_steps[1:])/1000000,
        np.array(running_avg_dynamic)*1000,
        marker_list[0],
        alpha=0.4, markersize=8,
        label='dynamic', color=color_list[0])

    ax.plot(np.array(time_steps[1:])/1000000,
        np.array(running_avg_static_64)*1000,
        marker_list[1],
        alpha=0.4, markersize=8,
        label='static-64', color=color_list[1])


    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)

    ax.legend(loc='upper right', prop=font, edgecolor="black", fancybox=False)

    ax.set_ylabel('Combined time (ms)', fontsize=FONTSIZE, font=FONT)
    ax.set_xlabel('Training steps (millions)', fontsize=FONTSIZE, font=FONT)
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 8])

    # axs[3, 1].set_xlabel('Steps (millions)', fontsize=12, font=FONT)
    # plt.style.use(["science", "grid"])

    plt.tight_layout()

    plt.savefig(
        f'/home/dts/Documents/theory/batching_paper/figs/time_series_batch_size_32_qm9_MPEU.png',
        dpi=600)
    plt.show()


def running_average(data_col, time_steps):
    """Get a running average of the mean combined times."""
    previous_time = 0
    running_average_list = []
    for time in time_steps[1:]:
        print(f'time is {time} and previous time is {previous_time}')
        running_average_list.append(np.mean(data_col[previous_time:int(time)]))
        previous_time = int(time)
    return running_average_list


def main(args):

    # plot_curves()
    plot_running_average_curves()



if __name__ == '__main__':
    app.run(main)

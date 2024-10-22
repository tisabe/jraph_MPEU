import argparse
import os
import pickle
from absl import flags
from absl import app
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'csv_filename',
    'None',
    'Where to store data as csv that has been parsed.')


# BATCH_SIZE_LIST = [16, 32, 64, 128]
BATCH_SIZE_LIST = [16, 32, 64, 128]

# MODEL_TYPE_LIST = ['schnet', 'mpeu']
MODEL_TYPE_LIST = ['schnet', 'MPEU']

# BATCH_METHOD_LIST = ['dynamic', 'static']
BATCH_METHOD_LIST = ['dynamic', 'static']
# COMPUTING_TYPE_LIST = ['gpu_a100', 'cpu']
COMPUTING_TYPE_LIST = ['gpu_a100']

DATASET_LIST = ['aflow']

DEFAULT_LONGER_BATCHING_CSV = "/home/dts/Documents/hu/jraph_MPEU/batch_data"

TRAINING_STEP = [
    '100', '200', '300', '400', '500', '600', '700', '800', '900', '1_000',
    '1_100', '1_200', '1_300', '1_400', '1_500', '1_600', '1_700', '1_800',
    '1_900', '2_000']

BATCH_SIZE_DICT = {
    '16': 0,
    '32': 1,
    '64': 2,
    '128': 3,
}

def get_column_list(data_split):
    """Get a list of columns to keep in the dataframe."""
    col_list = []
    for step in TRAINING_STEP:
        col_list.append('step_X_000_split_rmse'.replace('X', step).replace(
            'split', data_split))
    return col_list


def get_avg_std_rmse(
        df, model, batch_method, compute_type, batch_size, dataset, data_split):
    
    columns_to_keep = get_column_list(data_split)
    # print(columns_to_keep)
    # Keep only the training columns we want to keep.
    # df = df[[c for c in df.columns if c in columns_to_keep]]
    df[columns_to_keep]
    # Now keep only the rows we want.
    df = df[
        (df['model'] == model) & (df['batching_type'] == batch_method)
        & (df['computing_type'] == compute_type) & (df['batch_size'] == batch_size)
        & (df['dataset'] == dataset)]
    # Now get the standard devation across different rows.
    # return df.mean(skipna = True), df.std(skipna = True)
    return df.mean(), df.std()


def plot_curves(
        csv_file, model_types, batch_size_list, batch_method_list,
        compute_type, dataset, data_split='training'):
    # load the dataframe
    df = pd.read_csv(csv_file)
    fig, axs = plt.subplots(4, 2)
    for model in model_types:
        if model == 'schnet':
            y_shift = 1
        else:
            y_shift = 0
        print(f'y_shift is {y_shift}')
        for batch_method in batch_method_list:
            if batch_method == 'dynamic':
                point_style = 'x'
                markerfacecolor = 'blue'
            else:
                point_style = 'o'
                markerfacecolor = 'red'

            for batch_size in batch_size_list:
                avg_rmse_df, std_rmse_df = get_avg_std_rmse(
                    df, model, batch_method, compute_type, batch_size, dataset,
                    data_split)
                x_shift = BATCH_SIZE_DICT[str(batch_size)]
                print(f'x_shift is {x_shift}')
                # Ok we now have a df with multiple columns and a single value
                # we need to convert the columns into steps.
                step_list = [int(step) for step in TRAINING_STEP]
                avg_metric_list = []
                std_metric_list = []
                for col in get_column_list(data_split):
                    avg_metric_list.append(avg_rmse_df[col])
                    std_metric_list.append(std_rmse_df[col])
                # plt.plot(step_list, avg_metric_list, 'o')

                print(
                    f'batch_size: {batch_size}, batching_method: {batch_method},'
                    f' model: {model}')
                print(avg_metric_list)    

                axs[x_shift, y_shift].plot(step_list, avg_metric_list,
                                          marker=point_style,
                                          linestyle='dashed',
                                          markerfacecolor=markerfacecolor)
                
                # errorbar(step_list, avg_metric_list,
                #                 std_metric_list)
    plt.show()

def main(args):
    # plot learning curves
    fig, ax = plt.subplots(2)
    rmse_all = []
    mae_all = []
    # COMPUTING_TYPE_LIST
    plot_curves(
        FLAGS.csv_filename, MODEL_TYPE_LIST, BATCH_SIZE_LIST, BATCH_METHOD_LIST,
        COMPUTING_TYPE_LIST[0], DATASET_LIST[0], data_split='test')

    # try:
    #     metrics_path = args.file
    #     #metrics_path = 'results/mp/cutoff/lowlr/checkpoints/metrics.pkl'
    #     with open(metrics_path, 'rb') as metrics_file:
    #         metrics_dict = pickle.load(metrics_file)
    #     print(metrics_path)

    #     split = 'validation'
    #     metrics = metrics_dict[split]
    #     loss_rmse = [row[1][0] for row in metrics]
    #     loss_mae = [row[1][1] for row in metrics]
    #     step = [int(row[0]) for row in metrics]
    #     min_step_index = np.argmin(loss_rmse)
    #     print(f'Step with minimum loss: {step[min_step_index]}')
    #     # TODO: import config and show hyperparameters
    #     ax[0].plot(step, loss_rmse, label=metrics_path)
    #     ax[1].plot(step, loss_mae, label=metrics_path)

    #     #ax[0].legend()
    #     #ax[1].legend()
    #     ax[0].set_xlabel('gradient step', fontsize=12)
    #     ax[1].set_xlabel('gradient step', fontsize=12)
    #     ax[0].set_ylabel('RMSE (eV)', fontsize=12)
    #     ax[1].set_ylabel('MAE (eV)', fontsize=12)
    #     ax[0].set_yscale('log')
    #     ax[1].set_yscale('log')

    #     split = 'test'
    #     metrics = metrics_dict[split]
    #     loss_rmse = [row[1][0] for row in metrics]
    #     loss_mae = [row[1][1] for row in metrics]

    #     min_rmse = loss_rmse[min_step_index]
    #     min_mae = loss_mae[min_step_index]
    #     rmse_all.append(min_rmse)
    #     mae_all.append(min_mae)
    #     print(f'Minimum test RMSE: {min_rmse}')
    #     print(f'Minimum test MAE: {min_mae}')

    # except OSError:
    #     print(f'{metrics_path} not a valid path, path is skipped.')
    #     pass

    # #ax[0].legend()
    # #ax[1].legend()
    # print(f'Average RMSE: {np.mean(rmse_all)} +- {np.std(rmse_all)}')
    # print(f'Average MAE: {np.mean(mae_all)} +- {np.std(mae_all)}')
    # ax[0].set_xlabel('gradient step', fontsize=12)
    # ax[1].set_xlabel('gradient step', fontsize=12)
    # ax[0].set_ylabel('MSE (eV)', fontsize=12)
    # ax[1].set_ylabel('MAE (eV)', fontsize=12)
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    # plt.tight_layout()

    # plt.show()
    # fig.savefig(args.file+'/curves.png', bbox_inches='tight', dpi=600)

    # return 0


if __name__ == '__main__':
    app.run(main)

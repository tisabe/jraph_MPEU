import argparse
import os
import pickle
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def min_of_previous(array):
    return [min(array[:i]) for i in range(len(array))]



def main(args):
    # plot learning curves
    df = pd.DataFrame({})
    dict_minima = {}
    print(args.max_step)
    for dirname in os.listdir(args.file):
        try:
            metrics_path = args.file + '/'+dirname+'/checkpoints/metrics.pkl'
            # open the file with evaluation metrics
            with open(metrics_path, 'rb') as metrics_file:
                metrics_dict = pickle.load(metrics_file)

            config_path = args.file + '/' + dirname + '/config.json'
            with open(config_path, 'r') as config_file:
                config_dict = json.load(config_file)

            split = 'validation'
            metrics = metrics_dict[split]
            # get arrays with mae and rmse for this run
            loss_rmse = [row[1][0] for row in metrics if int(row[0]) < args.max_step]
            loss_mae = [row[1][1] for row in metrics if int(row[0]) < args.max_step]
            n_mean = 1 # number of points for running mean
            #  compute running mean using convolution
            loss_rmse = np.convolve(loss_rmse, np.ones(n_mean)/n_mean, mode='valid')
            loss_mae = np.convolve(loss_mae, np.ones(n_mean)/n_mean, mode='valid')
            min_mae = min(loss_mae)
            min_rmse = min(loss_rmse)
            if min_mae > 1e4 or min_rmse > 1e4:
                print(f'mae or rmse too high for path {dirname}')
                continue

            dict_minima[dirname] = list(loss_mae)
            step = [int(row[0]) for row in metrics if int(row[0]) < args.max_step]
            min_step_rmse = step[np.argmin(loss_rmse)]
            min_step_mae = step[np.argmin(loss_mae)]
            row_dict = {
                'mp_steps': int(config_dict['message_passing_steps']),
                'latent_size': int(config_dict['latent_size']),
                'init_lr': config_dict['init_lr'],
                'decay_rate': config_dict['decay_rate'],
                'dropout_rate': config_dict['dropout_rate'],
                'seed': config_dict['seed'],
                'mae': min_mae,
                'rmse': min_rmse,
                'min_step_mae': min_step_mae,
                'min_step_rmse': min_step_rmse,
                'directory': dirname
            }
            #print(row_dict)
            df = df.append(row_dict, ignore_index=True)

        except OSError:
            pass
            #print(f'{dirname} not a valid path, path is skipped.')

    dict_minima_top = {}

    # print the best 5 configs
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by='rmse', axis='index')
    for i in range(5):
        print(f'{i}. minimum rmse configuration: \n', df_copy.iloc[i])

    # print the worst 5 configs
    df_copy = df_copy.sort_values(by='rmse', axis='index', ascending=False)
    for i in range(5):
        print(f'{i}. maximum rmse configuration: \n', df_copy.iloc[i])
    """
    for i in range(10):
        # get index for lowest mae
        i_min = df_copy['rmse'].idxmin()
        print(f'{i}. minimum rmse configuration: \n', df_copy.iloc[i_min])
        name = df_copy.iloc[i_min]['directory']
        df_copy = df_copy.drop([i_min])
        dict_minima_top[name] = dict_minima[name]
    """
    """
    for column in dict_minima_top:
        plt.plot(dict_minima_top[column], label=column)
    plt.legend()
    plt.show()
    """
    # drop the worst 10 configs
    for i in range(10):
        i_max = df['rmse'].idxmax()
        df = df.drop([i_max])

    # plot rmse for main hyperparameters with logscale
    #box_xnames = ['latent_size', 'mp_steps', 'init_lr', 'decay_rate']
    #box_xnames = ['seed', 'dropout_rate']
    n_unique = df.nunique()
    n_dropped = n_unique.drop(n_unique[n_unique < 2].index)
    n_dropped = n_dropped.drop(
        labels=['mae', 'rmse', 'min_step_mae', 'min_step_rmse', 'directory'])
    print(n_dropped)
    box_xnames = list(n_dropped.keys())
    col_to_label = {
        'latent_size': 'Latent size', 'mp_steps': 'MP steps',
        'init_lr': 'Learning rate', 'decay_rate': 'LR decay rate',
        'dropout_rate': 'Dropout rate', 'seed': 'Split seed'}
    df = df.astype({'latent_size': 'int32'})
    df = df.astype({'mp_steps': 'int32'})
    df = df.astype({'seed': 'int32'})
    fig, ax = plt.subplots(1, len(box_xnames), figsize=(16, 8), sharey=True)
    for i, name in enumerate(box_xnames):
        sns.boxplot(ax=ax[i], x=name, y='rmse', data=df, color='C0')
        sns.swarmplot(ax=ax[i], x=name, y='rmse', data=df, color='.25')
        ax[i].set_xlabel(col_to_label[name], fontsize=22)
        if i == 0:
            ax[i].set_ylabel('RMSE (eV/atom)', fontsize=22)
        else:
            ax[i].set_ylabel('')
        ax[i].tick_params(axis='both', which='both', labelsize=18)
    #plt.yscale('log')
    plt.rc('font', size=16)
    plt.tight_layout()
    plt.show()
    fig.savefig(args.file+'/grid_search.png', bbox_inches='tight', dpi=600)


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show ensemble of loss curves.')
    parser.add_argument(
        '-f', '-F', type=str, dest='file',
        default='results/aflow/crossval_grid',
        help='input super directory name')
    parser.add_argument(
        '-step', type=int, dest='max_step',
        default=100000000,  # an arbitrary large number...
        help='maximum number of steps to take the mse/mae minimum from'
    )
    args_main = parser.parse_args()
    main(args_main)

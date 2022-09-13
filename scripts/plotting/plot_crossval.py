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
    df_minima = pd.DataFrame({})
    print(args.max_step)
    for dirname in os.listdir(args.file):
        try:
            metrics_path = args.file + '/'+dirname+'/checkpoints/metrics.pkl'
            #print(metrics_path)
            with open(metrics_path, 'rb') as metrics_file:
                metrics_dict = pickle.load(metrics_file)

            config_path = args.file + '/' + dirname + '/config.json'
            with open(config_path, 'r') as config_file:
                config_dict = json.load(config_file)

            split = 'validation'
            metrics = metrics_dict[split]
            loss_mse = [row[1][0] for row in metrics if int(row[0]) < args.max_step]
            loss_mae = [row[1][1] for row in metrics if int(row[0]) < args.max_step]
            n_mean = 5 # number of points for running mean
            #  compute running mean using convolution
            loss_mae = np.convolve(loss_mae, np.ones(n_mean)/n_mean, mode='valid')
            loss_mse = np.convolve(loss_mse, np.ones(n_mean)/n_mean, mode='valid')
            min_mae = min(loss_mae)
            min_mse = min(loss_mse)
            if min_mae > 1e4 or min_mse > 1e4:
                print(f'mae or mse too high for path {dirname}')
                continue

            df_minima[dirname] = list(loss_mae)
            step = [int(row[0]) for row in metrics if int(row[0]) < args.max_step]
            min_step_mse = step[np.argmin(loss_mse)]
            min_step_mae = step[np.argmin(loss_mae)]
            row_dict = {
                'mp_steps': config_dict['message_passing_steps'],
                'latent_size': config_dict['latent_size'],
                'init_lr': config_dict['init_lr'],
                'decay_rate': config_dict['decay_rate'],
                'mae': min_mae,
                'mse': min_mse,
                'min_step_mae': min_step_mae,
                'min_step_mse': min_step_mse,
                'directory': dirname
            }
            #print(row_dict)
            df = df.append(row_dict, ignore_index=True)

        except OSError:
            a = 1
            #print(f'{dirname} not a valid path, path is skipped.')

    df_minima_top = pd.DataFrame({})

    # print the best 10 configs
    df_copy = df.copy()
    for i in range(20):
        i_min = df_copy['mae'].idxmin()
        print(f'{i}. minimum mae configuration: \n', df_copy.iloc[i_min])
        name = df_copy.iloc[i_min]['directory']
        df_copy = df_copy.drop([i_min])
        df_minima_top[name] = df_minima[name]

    for column in df_minima_top:
        plt.plot(df_minima_top[column], label=column)
    plt.legend()
    plt.show()

    # drop the worst 50 configs
    for i in range(50):
        i_max = df['mae'].idxmax()
        df = df.drop([i_max])

    # plot mse for main hyperparameters with logscale
    box_xnames = ['latent_size', 'mp_steps', 'init_lr', 'decay_rate']
    col_to_label = {
        'latent_size': 'Latent size', 'mp_steps': 'MP steps',
        'init_lr': 'Learning rate', 'decay_rate': 'LR decay rate'}
    fig, ax = plt.subplots(1, len(box_xnames), figsize=(16, 8), sharey=True)
    for i, name in enumerate(box_xnames):
        sns.boxplot(ax=ax[i], x=name, y='mae', data=df)
        sns.swarmplot(ax=ax[i], x=name, y='mae', data=df, color='.25')
        ax[i].set_xlabel(col_to_label[name], fontsize=22)
        if i == 0:
            ax[i].set_ylabel('MAE (eV/atom)', fontsize=22)
        else:
            ax[i].set_ylabel('')
    plt.yscale('log')
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

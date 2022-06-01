import argparse
import os
import pickle
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(args):
    # plot learning curves
    df = pd.DataFrame({})
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
            loss_mse = [row[1][0] for row in metrics]
            loss_mae = [row[1][1] for row in metrics]
            n_mean = 5 # number of points for running mean
            #  compute running mean using convolution
            loss_mae = np.convolve(loss_mae, np.ones(n_mean)/n_mean, mode='valid')
            loss_mse = np.convolve(loss_mse, np.ones(n_mean)/n_mean, mode='valid')
            min_mae = min(loss_mae)
            min_mse = min(loss_mse)
            if min_mae > 1e4 or min_mse > 1e4:
                print(f'mae or mse too high for path {dirname}')
                continue
            step = [int(row[0]) for row in metrics]
            min_step_mse = step[np.argmin(loss_mse)]
            min_step_mae = step[np.argmin(loss_mae)]
            row_dict = {
                'mp_steps': config_dict['message_passing_steps'],
                'latent_size': config_dict['latent_size'],
                'batch_size': config_dict['batch_size'],
                'init_lr': config_dict['init_lr'],
                'decay_rate': config_dict['decay_rate'],
                'mae': min_mae,
                'mse': min_mse,
                'min_step_mae': min_step_mae,
                'min_step_mse': min_step_mse,
            }
            #print(row_dict)
            df = df.append(row_dict, ignore_index=True)

        except OSError:
            a = 1
            #print(f'{dirname} not a valid path, path is skipped.')

    #sns.pairplot(df, y_vars=['mae', 'mse', 'min_step_mae', 'min_step_mse'])
    #plt.show()

    # print the best and worst 3 configs
    for i in range(3):
        i_min = df['mae'].idxmin()
        i_max = df['mae'].idxmax()
        print(f'{i}. minimum mae configuration: \n', df.iloc[i_min])
        print(f'{i}. maximum mae configuration: \n', df.iloc[i_max])
        df = df.drop([i_min, i_max])

    # plot mse for main hyperparameters with logscale
    box_xnames = ['latent_size', 'mp_steps', 'batch_size', 'init_lr', 'decay_rate']
    fig, ax = plt.subplots(1, len(box_xnames), figsize=(16, 8), sharey=True)
    for i, name in enumerate(box_xnames):
        sns.boxplot(ax=ax[i], x=name, y='mae', data=df)
        sns.swarmplot(ax=ax[i], x=name, y='mae', data=df, color='.25')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(args.file+'/pairplot.png', bbox_inches='tight', dpi=600)


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show ensemble of loss curves.')
    parser.add_argument(
        '-f', '-F', type=str, dest='file',
        default='results/aflow/crossval_grid',
        help='input super directory name')
    args_main = parser.parse_args()
    main(args_main)

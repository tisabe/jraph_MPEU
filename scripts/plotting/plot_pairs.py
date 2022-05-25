import argparse
import os
import pickle
import json

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
            #metrics_path = 'results/mp/cutoff/lowlr/checkpoints/metrics.pkl'
            with open(metrics_path, 'rb') as metrics_file:
                metrics_dict = pickle.load(metrics_file)

            config_path = args.file + '/' + dirname + '/config.json'
            with open(config_path, 'r') as config_file:
                config_dict = json.load(config_file)

            split = 'validation'
            metrics = metrics_dict[split]
            loss_mse = [row[1][0] for row in metrics]
            loss_mae = [row[1][1] for row in metrics]
            step = [int(row[0]) for row in metrics]
            row_dict = {
                'mp_steps': config_dict['message_passing_steps'],
                'latent_size': config_dict['latent_size'],
                'batch_size': config_dict['batch_size'],
                'decay_rate': config_dict['decay_rate'],
                'mae': min(loss_mae),
                'mse': min(loss_mse)}
            #print(row_dict)
            df = df.append(row_dict, ignore_index=True)

        except OSError:
            a = 1
            #print(f'{dirname} not a valid path, path is skipped.')

    sns.pairplot(df)
    plt.show()

    # print the best and worst 3 configs
    for i in range(3):
        i_min = df['mae'].idxmin()
        i_max = df['mae'].idxmax()
        print(f'{i}. minimum mae configuration: \n', df.iloc[i_min])
        print(f'{i}. maximum mae configuration: \n', df.iloc[i_max])
        df = df.drop([i_min, i_max])

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show ensemble of loss curves.')
    parser.add_argument('-f', '-F', type=str, dest='file', default='results_test',
                        help='input super directory name')
    args_main = parser.parse_args()
    main(args_main)

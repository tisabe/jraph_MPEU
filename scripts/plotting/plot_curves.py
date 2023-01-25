import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    # plot learning curves
    fig, ax = plt.subplots(2)
    rmse_all = []
    mae_all = []
    for dirname in os.listdir(args.file):
        try:
            metrics_path = args.file + '/'+dirname+'/checkpoints/metrics.pkl'
            print(metrics_path)
            #metrics_path = 'results/mp/cutoff/lowlr/checkpoints/metrics.pkl'
            with open(metrics_path, 'rb') as metrics_file:
                metrics_dict = pickle.load(metrics_file)

            split = 'validation'
            metrics = metrics_dict[split]
            loss_rmse = [row[1][0] for row in metrics]
            loss_mae = [row[1][1] for row in metrics]
            step = [int(row[0]) for row in metrics]
            # TODO: import config and show hyperparameters
            ax[0].plot(step, loss_rmse, label=metrics_path)
            ax[1].plot(step, loss_mae, label=metrics_path)

            #ax[0].legend()
            #ax[1].legend()
            ax[0].set_xlabel('gradient step', fontsize=12)
            ax[1].set_xlabel('gradient step', fontsize=12)
            ax[0].set_ylabel('RMSE (eV)', fontsize=12)
            ax[1].set_ylabel('MAE (eV)', fontsize=12)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')

            split = 'test'
            metrics = metrics_dict[split]
            loss_rmse = [row[1][0] for row in metrics]
            loss_mae = [row[1][1] for row in metrics]
            min_rmse = min(loss_rmse)
            min_mae = min(loss_mae)
            rmse_all.append(min_rmse)
            mae_all.append(min_mae)
            print(f'Minimum test RMSE: {min_rmse}')
            print(f'Minimum test MAE: {min_mae}')

        except OSError:
            print(f'{dirname} not a valid path, path is skipped.')

    #ax[0].legend()
    #ax[1].legend()
    print(f'Average RMSE: {np.mean(rmse_all)} +- {np.std(rmse_all)}')
    print(f'Average MAE: {np.mean(mae_all)} +- {np.std(mae_all)}')
    ax[0].set_xlabel('gradient step', fontsize=12)
    ax[1].set_xlabel('gradient step', fontsize=12)
    ax[0].set_ylabel('MSE (eV)', fontsize=12)
    ax[1].set_ylabel('MAE (eV)', fontsize=12)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    plt.tight_layout()

    plt.show()
    fig.savefig(args.file+'/curves.png', bbox_inches='tight', dpi=600)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show ensemble of loss curves.')
    parser.add_argument('-f', '-F', type=str, dest='file', default='results_test',
                        help='input super directory name')
    args = parser.parse_args()
    main(args)

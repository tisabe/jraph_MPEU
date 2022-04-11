import argparse

import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
import pickle


def plot_curve(metrics_dict, splits, folder):
    fig, ax = plt.subplots()

    for split in splits:
        loss = np.array(metrics_dict[split])
        ax.plot(loss[:, 0], loss[:, 1], label=split)      

    ax.legend()
    ax.set_xlabel('gradient step')
    ax.set_ylabel('MSE (eV^2), standardized')
    plt.yscale('log')
    plt.show()
    plt.savefig(folder+'/curve.png')


def main(args):
    folder = args.file

    splits = ['train', 'validation', 'test']

    # plot learning curves
    try:
        metrics_path = folder+'/checkpoints/metrics.pkl'
        #metrics_path = 'results/mp/cutoff/lowlr/checkpoints/metrics.pkl'
        with open(metrics_path, 'rb') as metrics_file:
            metrics_dict = pickle.load(metrics_file)
        
        fig, ax = plt.subplots()

        for split in splits:
            metrics = metrics_dict[split]
            loss = [row[1][1] for row in metrics]
            step = [int(row[0]) for row in metrics]
            ax.plot(step, loss, label=split)

        ax.legend()
        ax.set_xlabel('gradient step')
        ax.set_ylabel('MAE (eV), standardized')
        plt.yscale('log')
        plt.show()
        plt.savefig(folder+'/curve.png')
    except FileNotFoundError:
        print(f'Did not find {metrics_path}, skipping curve plot.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Show regression plot and loss curve.')
    parser.add_argument(
        '-f', '-F', type=str, dest='file', default='results/mp/cutoff/lowlr',
        help='input directory name')
    args = parser.parse_args()
    main(args)
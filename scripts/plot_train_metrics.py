"""Plot the train metrics saved after evaluating during the training.

Plot MSE and MAE. In training MSE is optimized, but MSE will be the final
metric. Both metrics have been scaled back to the real un-normalized value,
using the standard deviation of the labels."""

import argparse
import pickle

import matplotlib.pyplot as plt


def main(args_parsed):
    """Plot training curves."""
    folder = args_parsed.file

    splits = ['train', 'validation', 'test']

    # plot learning curves
    try:
        metrics_path = folder+'/checkpoints/metrics.pkl'
        #metrics_path = 'results/mp/cutoff/lowlr/checkpoints/metrics.pkl'
        with open(metrics_path, 'rb') as metrics_file:
            metrics_dict = pickle.load(metrics_file)

        _, ax = plt.subplots(2)

        for split in splits:
            metrics = metrics_dict[split]
            loss_mse = [row[1][0] for row in metrics]
            loss_mae = [row[1][1] for row in metrics]
            step = [int(row[0]) for row in metrics]
            ax[0].plot(step, loss_mae, label=split)
            ax[1].plot(step, loss_mse, label=split)

        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel('gradient step')
        ax[1].set_xlabel('gradient step')
        ax[0].set_ylabel('MSE (eV)')
        ax[1].set_ylabel('MAE (eV)')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
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

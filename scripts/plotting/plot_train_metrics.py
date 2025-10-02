"""Plot the train metrics saved after evaluating during the training.

Plot MSE and MAE. In training MSE is optimized, but MSE will be the final
metric. Both metrics have been scaled back to the real un-normalized value,
using the standard deviation of the labels."""

import pickle

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_integer('font_size', 12, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 12, 'font size to use in labels')
flags.DEFINE_string('loss_primary', 'RMSE (eV)', 'Name for primary loss label')
flags.DEFINE_string(
    'loss_secondary', 'MAE (eV)', 'Name for secondary loss label')
ABS_ERROR_LABEL = ''


def main(args_parsed):
    """Plot training curves."""
    folder = FLAGS.file

    splits = ['train', 'validation', 'test']
    split_convert = {
        'train': 'Training', 'validation': 'Validation', 'test': 'Test'}

    # plot learning curves
    try:
        metrics_path = folder+'/checkpoints/metrics.pkl'
        #metrics_path = 'results/mp/cutoff/lowlr/checkpoints/metrics.pkl'
        with open(metrics_path, 'rb') as metrics_file:
            metrics_dict = pickle.load(metrics_file)

        fig, ax = plt.subplots(2, sharex=True)

        for split in splits:
            metrics = metrics_dict[split]
            loss_mse = [row[1][0] for row in metrics]
            loss_mae = [row[1][1] for row in metrics]
            step = [int(row[0]) for row in metrics]
            ax[0].plot(step, loss_mse, label=split_convert[split])
            ax[1].plot(step, loss_mae, label=split_convert[split])

        ax[1].legend(fontsize=FLAGS.font_size-3)
        ax[1].set_xlabel('Gradient step', fontsize=FLAGS.font_size)
        ax[0].set_ylabel(FLAGS.loss_primary, fontsize=FLAGS.font_size)
        ax[1].set_ylabel(FLAGS.loss_secondary, fontsize=FLAGS.font_size)
        if np.all(np.asarray(loss_mse)>0):
            ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].xaxis.set_major_formatter(ticker.EngFormatter())
        ax[0].tick_params(which='both', labelsize=FLAGS.tick_size)
        ax[1].tick_params(which='both', labelsize=FLAGS.tick_size)
        plt.tight_layout()

        plt.show()
        fig.savefig(folder+'/curve.png', bbox_inches='tight', dpi=600)
    except FileNotFoundError:
        print(f'Did not find {metrics_path}, skipping curve plot.')


if __name__ == "__main__":
    app.run(main)

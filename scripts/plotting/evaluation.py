"""Evaluate the model using checkpoints and best state from workdir.

The model with the best validation loss is saved during training and loaded
here. The model weights are saved in the pickle file, after they are loaded,
the model can be built using the config.json.
"""

import argparse

from absl import logging
import matplotlib.pyplot as plt
import numpy as np

from jraph_MPEU.inference import load_inference_file


def main(args):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    workdir = args.folder

    inference_dict = load_inference_file(workdir, redo=args.redo)

    fig, ax = plt.subplots(2)
    marker_size = 0.3

    splits = inference_dict.keys()
    units = input("Type units of prediction and target: ")
    for split in splits:
        preds = inference_dict[split]['preds']
        targets = inference_dict[split]['targets']
        error = np.abs(preds - targets)
        mae = np.mean(error)
        mse = np.mean(np.square(error))
        print(f'Number of graphs: {len(preds)}')
        print(f'MSE: {mse} {units}')
        print(f'MAE: {mae} {units}')
        label_string = f'{split} \nMAE: {mae:9.3f} {units}'
        ax[0].scatter(targets, preds, s=marker_size, label=label_string)
        ax[1].scatter(targets, error, s=marker_size, label=split)

    # plot x = y regression lines
    x_ref = np.linspace(*ax[0].get_xlim())
    ax[0].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[0].set_title('Model regression performance')
    ax[0].set_ylabel(f'prediction ({units})', fontsize=12)
    ax[1].set_xlabel(f'target ({units})', fontsize=12)
    ax[1].set_ylabel(f'abs. error ({units})', fontsize=12)
    ax[0].set_aspect('equal')
    ax[0].legend()
    ax[1].legend()
    ax[1].set_yscale('log')
    plt.tight_layout()

    plt.show()
    fig.savefig(workdir+'/fit.png', bbox_inches='tight', dpi=600)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Show regression plot and loss curve.')
    parser.add_argument(
        '-f', '-F', type=str, dest='folder', default='results/qm9/test',
        help='input directory name')
    parser.add_argument(
        '--redo', dest='redo', action='store_true'
    )
    args_main = parser.parse_args()
    main(args_main)

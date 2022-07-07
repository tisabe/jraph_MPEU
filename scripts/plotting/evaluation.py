"""Evaluate the model using checkpoints and best state from workdir.

The model with the best validation loss is saved during training and loaded
here. The model weights are saved in the pickle file, after they are loaded,
the model can be built using the config.json.
"""

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np

from jraph_MPEU.inference import load_inference_file

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')

def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    workdir = FLAGS.file

    inference_dict = load_inference_file(workdir, redo=FLAGS.redo)

    fig, ax = plt.subplots()
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
        ax.scatter(targets, preds, s=marker_size, label=label_string)

    # plot x = y regression lines
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax.set_title('Model regression performance')
    ax.set_ylabel(f'prediction ({units})', fontsize=12)
    ax.set_xlabel(f'target ({units})', fontsize=12)
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()

    plt.show()
    fig.savefig(workdir+'/fit.png', bbox_inches='tight', dpi=600)

    ### plot residuals
    fig, ax = plt.subplots()
    for split in splits:
        preds = inference_dict[split]['preds']
        targets = inference_dict[split]['targets']
        error = (preds - targets)
        mae = np.mean(error)
        mse = np.mean(np.square(error))
        print(f'Number of graphs: {len(preds)}')
        print(f'MSE: {mse} {units}')
        print(f'MAE: {mae} {units}')
        label_string = split
        ax.scatter(targets, error, s=marker_size, label=label_string)

    ax.set_title('Model regression performance')
    ax.set_ylabel(f'error ({units})', fontsize=12)
    ax.set_xlabel(f'target ({units})', fontsize=12)
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()

    plt.show()
    fig.savefig(workdir+'/residuals.png', bbox_inches='tight', dpi=600)



if __name__ == "__main__":
    app.run(main)

"""Evaluate the model using checkpoints and best state from workdir.

The model with the best validation loss is saved during training and loaded
here. The model weights are saved in the pickle file, after they are loaded,
the model can be built using the config.json.
"""

from absl import app
from absl import flags
from absl import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jraph_MPEU.inference import load_inference_file

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')

#PREDICT_LABEL = 'Predicted formation energy (eV/atom)'
#PREDICT_LABEL = r'Predicted $U_0$ (eV)'
PREDICT_LABEL = r'Predicted band gap $(eV)$'
#CALCULATE_LABEL = 'Calculated formation energy (eV/atom)'
#CALCULATE_LABEL = r'Calculated $U_0$ (eV)'
CALCULATE_LABEL = r'Calculated band gap $(eV)$'
RESIDUAL_LABEL = r'Residual $U_0^{Model} - U_0^{Target}$ (eV)'
ABS_ERROR_LABEL = 'MAE (eV)'

def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    workdir = FLAGS.file

    inference_dict = load_inference_file(workdir, redo=FLAGS.redo)

    df = pd.DataFrame({})
    df['prediction'] = inference_dict['test']['preds']
    df['target'] = inference_dict['test']['targets']

    plt.hist2d(df['target'], df['prediction'], bins=100, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots()
    sns.histplot(
        x='target',  # plot prediction vs label
        y='prediction',
        data=df,
        #hue='split',
        cbar=True, cbar_kws={'label': 'Count'},
        ax=ax,
        bins=(100, 100),
        #norm=mpl.colors.LogNorm()
    )
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax.set_xlabel(CALCULATE_LABEL, fontsize=12)
    ax.set_ylabel(PREDICT_LABEL, fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/fit.png', bbox_inches='tight', dpi=600)

    df['residual'] = df['prediction'] - df['target']
    print(df)
    fig, ax = plt.subplots()
    sns.histplot(
        x='target',  # plot prediction vs label
        y='residual',
        data=df,
        #hue='split',
        cbar=True, cbar_kws={'label': 'Count'},
        ax=ax,
    )
    ax.set_xlabel(CALCULATE_LABEL, fontsize=12)
    ax.set_ylabel(RESIDUAL_LABEL, fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/residuals.png', bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    app.run(main)

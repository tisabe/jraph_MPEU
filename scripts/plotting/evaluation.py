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
flags.DEFINE_string('label', 'ef', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "ef" or "egap"')
flags.DEFINE_integer('font_size', 12, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 12, 'font size to use in labels')

#PREDICT_LABEL = 'Predicted formation energy (eV/atom)'
PREDICT_LABEL = r'Predicted $U_0$ (eV)'
#PREDICT_LABEL = r'Predicted band gap $(eV)$'
#CALCULATE_LABEL = 'Calculated formation energy (eV/atom)'
CALCULATE_LABEL = r'Calculated $U_0$ (eV)'
#CALCULATE_LABEL = r'Calculated band gap $(eV)$'
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
    len_dim0 = len(inference_dict['test']['preds'])
    print(np.shape(inference_dict['test']['preds']))
    df['prediction'] = inference_dict['test']['preds'] if len_dim0 > 1 else inference_dict['test']['preds'][0]
    df['target'] = inference_dict['test']['targets'] if len_dim0 > 1 else inference_dict['test']['targets'][0]

    plt.hist2d(df['target'], df['prediction'], bins=100, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))
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
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 14
    cbar.ax.tick_params(labelsize=14)
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax.set_xlabel(CALCULATE_LABEL, fontsize=FLAGS.font_size)
    ax.set_ylabel(PREDICT_LABEL, fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/fit.png', bbox_inches='tight', dpi=600)

    df['residual'] = df['prediction'] - df['target']
    print(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        x='target',  # plot prediction vs label
        y='residual',
        data=df,
        #cbar=True, cbar_kws={'label': 'Count'},
        ax=ax,
        alpha=0.4
    )
    ax.set_xlabel(CALCULATE_LABEL, fontsize=12)
    ax.set_ylabel(RESIDUAL_LABEL, fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/residuals.png', bbox_inches='tight', dpi=600)



if __name__ == "__main__":
    app.run(main)

"""This script produces plots to evaluate the errors of the model inferences,
using different metrics such as atomic numbers, number of species etc.
"""
import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from jraph_MPEU.utils import load_config
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')

def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    workdir = FLAGS.file
    df_path = workdir + '/result.csv'

    if not os.path.exists(df_path) or FLAGS.redo:
        logging.info('Did not find csv path, generating DataFrame.')
        df = get_results_df(workdir)
        df.head()
        print(df)
        df.to_csv(df_path, index=False)
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df = pd.read_csv(df_path)

    fig, ax = plt.subplots()
    sns.scatterplot(x='enthalpy_formation_atom', y='prediction', data=df, hue='split', ax=ax)
    plt.show()
    fig.savefig(workdir+'/error_analysis.png', bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    app.run(main)

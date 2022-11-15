"""Script to visually analyze classification results from a classifier model."""
import os

from absl import app
from absl import flags
from absl import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

from jraph_MPEU.utils import load_config, str_to_list
from jraph_MPEU.inference import get_results_df
from jraph_MPEU.input_pipeline import cut_egap

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')
flags.DEFINE_integer('font_size', 12, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 12, 'font size to use in labels')


def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    workdir = FLAGS.file
    df_path = workdir + '/result.csv'
    config = load_config(workdir)

    if not os.path.exists(df_path) or FLAGS.redo:
        logging.info('Did not find csv path, generating DataFrame.')
        df = get_results_df(workdir, FLAGS.limit)
        df.head()
        print(df)
        df.to_csv(df_path, index=False)
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df = pd.read_csv(df_path)
        df['numbers'] = df['numbers'].apply(str_to_list)

    df['class_true'] = df['Egap'].apply(cut_egap) #  1 is insulator, 0 is metal
    df['p_insulator'] = 1 - df['prediction']
    df['class_pred'] = df['p_insulator'].apply(lambda p: (p > 0.5)*1)
    #print(df[['class_true', 'p_insulator']])
    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']

    plt.hist(df_test['p_insulator'], bins=100, log=True)
    #plt.show()
    # calculate binary cross entropy
    #bce = -1 * np.array(df['class_true']) * np.log(df['p_insulator']) \
    #    - (1 - np.array(df['class_true'])) * np.log(1 - np.array(df['p_insulator']))
    #plt.hist(bce)

    print(df_test['class_pred'][df_test['class_pred'] == 1])

    # calculate and display ROC curve
    y_pred = df_test['p_insulator'].to_numpy().reshape(-1, 1)
    y_true = df_test['class_true'].to_numpy()
    fig, ax = plt.subplots()
    display = RocCurveDisplay.from_predictions(y_true, y_pred, ax=ax)
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax.set_xlabel('False positive rate', fontsize=FLAGS.font_size)
    ax.set_ylabel('True positive rate', fontsize=FLAGS.font_size)
    ax.legend(title='', fontsize=FLAGS.font_size-5)
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/roc_curve.png', bbox_inches='tight', dpi=600)

    # print ROC-AUC score
    auc = roc_auc_score(y_true, y_pred)
    print(f"ROC-AUC score on test set: {auc}")

    acc = sklearn.metrics.accuracy_score(
        df_test['class_true'], df_test['class_pred'])
    print(f"Total accuracy on test set: {acc}")

if __name__ == "__main__":
    app.run(main)

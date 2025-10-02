"""Analyze the regression performance of a electronic band-gap prediction."""
import os

from absl import app
from absl import flags
from absl import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve

from jraph_MPEU.utils import str_to_list
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
    """Get the model inferences and plot classification probabilities and
    scores like accuracy and ROC-AUC on the test set."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    workdir = FLAGS.file
    df_path = workdir + '/result.csv'

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

    df['abs. error'] = abs(df['prediction'] - df['Egap'])
    df['class_true'] = (df['Egap'] > 1)*1.
    df['p_insulator'] = df['prediction'].clip(lower=0., upper=1.)

    df_val = df.loc[lambda df_temp: df_temp['split'] == 'validation']

    y_pred_val = df_val['p_insulator'].to_numpy().reshape(-1, 1)
    y_true_val = df_val['class_true'].to_numpy()
    fpr, tpr, thresholds = roc_curve(y_true_val, y_pred_val)
    i_opt = np.argmax(tpr - fpr)
    print(f"Optimal threshold: {thresholds[i_opt]}")
    print(f"Optimal fpr: {fpr[i_opt]}")
    print(f"Optimal tpr: {tpr[i_opt]}")
    df['class_pred'] = (df['prediction'] > thresholds[i_opt])*1.

    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
    # calculate and display ROC curve
    y_pred = df_test['p_insulator'].to_numpy().reshape(-1, 1)
    #y_pred = np.random.random_sample(size=(len(df_test), 1)) # to sanity check with random values
    y_true = df_test['class_true'].to_numpy()

    auc = roc_auc_score(y_true, y_pred)
    print(f"ROC-AUC score on test set: {auc}")

    fig, ax = plt.subplots()
    _ = RocCurveDisplay.from_predictions(y_true, y_pred, ax=ax)
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax.set_xlabel('False positive rate', fontsize=FLAGS.font_size)
    ax.set_ylabel('True positive rate', fontsize=FLAGS.font_size)
    ax.legend(title='', fontsize=FLAGS.font_size-5)
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/roc_curve.png', bbox_inches='tight', dpi=600)

    print('Confusion matrix (test split): ')
    cf_matrix = sklearn.metrics.confusion_matrix(
        y_true=df_test['class_true'], y_pred=df_test['class_pred'])
    print(cf_matrix)
    print("[[TN\tFP],\n [FN\tTP]]")
    print(sklearn.metrics.classification_report(
        y_true=df_test['class_true'], y_pred=df_test['class_pred'],
        target_names=['Metals','Non-metals']
    ))

    df_test_ins = df_test.loc[lambda df_temp: df_temp['class_pred'] == 1]
    mean_abs_err_test = df_test_ins['abs. error'].mean()
    mse_test = (df_test_ins['abs. error']**2).mean()
    mdae_test = df_test_ins['abs. error'].median()
    print(f"RMSE (test split, predicted insulators): {np.sqrt(mse_test)}")
    print(f"MAE (test split, predicted insulators): {mean_abs_err_test}")
    print(f"MdAE (test split, predicted insulators): {mdae_test}")


if __name__ == "__main__":
    app.run(main)

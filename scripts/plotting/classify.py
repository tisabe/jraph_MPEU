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
from sklearn.calibration import CalibrationDisplay

from jraph_MPEU.utils import str_to_list
from jraph_MPEU.inference import get_results_df
from jraph_MPEU.input_pipeline import cut_egap

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')
flags.DEFINE_integer('font_size', 12, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 12, 'tick size to use in labels')


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

    if not 'prediction' in df.columns:
        df['prediction'] = df['prediction_mean']

    df['class_true'] = df['Egap'].apply(cut_egap) #  1 is insulator, 0 is metal
    df['p_insulator'] = 1 - df['prediction']
    # calculate the class prediction by applying a threshold. Because of the
    # softmax outputs probability, the threshold is exactly 1/2
    df['class_pred'] = df['p_insulator'].apply(lambda p: (p > 0.5)*1)
    df['class_correct'] = df['class_true'] == df['class_pred']
    df['Egap_bin'] = pd.cut(df['Egap'], 20)
    print(df['Egap_bin'])

    print(df[df['class_correct'] == False][[
        'class_true', 'class_pred', 'class_correct']])
    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
    # calculate confusion matrix on test split
    print('Confusion matrix (test split): ')
    cf_matrix = sklearn.metrics.confusion_matrix(
        y_true=df_test['class_true'], y_pred=df_test['class_pred'])
    print(cf_matrix)
    print("[[TN\tFP],\n [FN\tTP]]")
    print(sklearn.metrics.classification_report(
        y_true=df_test['class_true'], y_pred=df_test['class_pred'],
        target_names=['Metals','Non-metals']
    ))
    print('Stats for whole dataset: ')
    print(sklearn.metrics.classification_report(
        y_true=df['class_true'], y_pred=df['class_pred'],
        target_names=['Metals','Non-metals']
    ))

    # calculate and display ROC curve
    y_pred = df_test['p_insulator'].to_numpy().reshape(-1, 1)
    y_true = df_test['class_true'].to_numpy()

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

    # calculate binned egap plot
    data = df_test.groupby(by='Egap_bin')[['class_correct', 'Egap']].aggregate(
        ['mean', 'min', 'max', 'count'])
    print(data)
    fig, ax = plt.subplots()
    color = 'tab:blue'
    ax.bar(
        x=data['Egap']['min'],
        height=data['class_correct']['mean'],
        width=0.8,
        align='edge')
    ax.set_ylim([.9, 1.005])
    ax.set_ylabel('Accuracy', color=color, fontsize=FLAGS.font_size)
    ax.set_xlabel(r'$E_g$ (eV)', color='black', fontsize=FLAGS.font_size)
    ax.tick_params(
        axis='y', which='both', labelcolor=color, labelsize=FLAGS.tick_size)
    ax.tick_params(
        axis='x', which='both', labelcolor='black', labelsize=FLAGS.tick_size)

    color = 'tab:orange'
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.bar(
        x=data['Egap']['min'],
        height=data['Egap']['count'],
        width=0.5,
        align='edge',
        color=color)
    ax2.set_ylabel('Count', color=color, fontsize=FLAGS.font_size)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=FLAGS.tick_size)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/accuracy_binned.png', bbox_inches='tight', dpi=600)

    # display calibration curve
    fig, ax = plt.subplots()
    _ = CalibrationDisplay.from_predictions(y_true, y_pred, n_bins=10, ax=ax)
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()

    # print ROC-AUC score
    auc = roc_auc_score(y_true, y_pred)
    print(f"ROC-AUC score on test set: {auc}")

    acc = sklearn.metrics.accuracy_score(
        df_test['class_true'], df_test['class_pred'])
    print(f"Total accuracy on test set: {acc}")

    df_correct = df[df['class_correct'] == True]
    df_incorrect = df[df['class_correct'] == False]
    plt.hist(df_correct['p_insulator'], bins=100, log=True)
    plt.hist(df_incorrect['p_insulator'], bins=100, log=True)
    plt.xlabel(r'$\hat{p}_{insulator}$')
    plt.ylabel('count')
    #plt.show()
    #sns.histplot(df_test, x='p_insulator', hue='class_correct', log_scale=False)
    plt.show()

    counts_correct, bins_correct = np.histogram(df['p_insulator'], bins=100)
    counts_incorrect, _ = np.histogram(df_incorrect['p_insulator'], bins=100)
    plt.stairs(counts_incorrect/counts_correct, bins_correct, fill=True)
    plt.show()

    # plot distribution of band gaps of predicted insulators
    df_insulators_test = df_test[df_test['class_true'] == 1]
    print(df_insulators_test)
    sns.histplot(df_insulators_test, x='Egap', bins=100)
    plt.show()

if __name__ == "__main__":
    app.run(main)

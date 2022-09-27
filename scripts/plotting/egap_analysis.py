"""This script analyses energy band gap predictions on AFLOW data (for now)."""

import os
from collections import Counter

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

from jraph_MPEU.utils import load_config
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')


def plot_egap_hist(dataframe: pd.DataFrame, workdir):
    egaps = dataframe['Egap']
    egaps = egaps[egaps > 0]  # filter metals, to make log hist possible
    # histogram on linear scale
    ax1 = plt.subplot(221)
    hist, bins, _ = plt.hist(egaps, bins=100)
    plt.yscale('log')
    plt.xlabel('Target Egap (no metals)')

    # histogram on log scale.
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    #print(logbins)
    ax2 = plt.subplot(222)
    plt.hist(egaps, bins=logbins)
    #plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Target Egap (no metals)')

    preds = dataframe['prediction']
    preds = preds[preds > 0]
    # histogram on linear scale
    ax3 = plt.subplot(223)
    hist, bins, _ = plt.hist(preds, bins=100)
    plt.yscale('log')
    plt.xlabel('Predicted Egap (> 0)')

    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    fig, ax4 = plt.subplot(224, sharex=ax2)
    plt.hist(preds, bins=logbins)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Predicted Egap (> 0)')

    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/egap_hist.png', bbox_inches='tight', dpi=600)


def egap_type_to_class(egap_type: str):
    """Return class 0 or 1 for metal/half-metal or non-metal egap_type string."""
    if egap_type in ('metal', 'half-metal'):
        return 0
    else:
        return 1


def classify_egap(dataframe, workdir):
    dataframe['egap_class'] = dataframe['Egap_type'].apply(egap_type_to_class)
    print(dataframe[['Egap_type', 'egap_class']])
    # split up dataframe into train and test
    df_train = dataframe.loc[lambda df_temp: df_temp['split'] == 'train']
    df_val = dataframe.loc[lambda df_temp: df_temp['split'] == 'validation']
    df_test = dataframe.loc[lambda df_temp: df_temp['split'] == 'test']

    clf = LogisticRegression(random_state=0).fit(
        df_train['prediction'].to_numpy().reshape(-1, 1),
        df_train['egap_class'].to_numpy()
    )
    score = clf.score(
        df_train['prediction'].to_numpy().reshape(-1, 1),
        df_train['egap_class'].to_numpy()
    )
    print(f'Mean accuracy on train set: {score}')
    score = clf.score(
        df_val['prediction'].to_numpy().reshape(-1, 1),
        df_val['egap_class'].to_numpy()
    )
    print(f'Mean accuracy on validation set: {score}')
    score = clf.score(
        df_test['prediction'].to_numpy().reshape(-1, 1),
        df_test['egap_class'].to_numpy()
    )
    print(f'Mean accuracy on test set: {score}')

    # plot the egap vs prediction with prediction of the egap type
    df_test['egap_class_predict'] = clf.predict(
        df_test['prediction'].to_numpy().reshape(-1, 1)
    )
    df_test['correct_class'] = df_test['egap_class_predict'] == df_test['egap_class']
    df_test['correct_class'] = df_test['correct_class'].apply(
        lambda x: 'right classification' if x else 'wrong classification'
    )
    matrix = metrics.confusion_matrix(df_test['egap_class'], df_test['egap_class_predict'])
    print(matrix)
    print(matrix/np.sum(matrix))
    print(np.array([['TP', 'FN'], ['FP', 'TN']]))
    fig, ax = plt.subplots()
    sns.scatterplot(
        x='Egap',
        y='prediction',
        data=df_test,
        hue='correct_class',
        palette=('green', 'red'),
        ax=ax,
        alpha=0.2
    )
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax.set_xlabel(r'Target E$_{BG}$ (eV)', fontsize=12)
    ax.set_ylabel(r'Predicted E$_{BG}$ (eV)', fontsize=12)
    ax.set_box_aspect(1)
    ax.legend(title='')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/egap_classify.png', bbox_inches='tight', dpi=600)

    # calculate and display ROC curve
    x_test = df_test['prediction'].to_numpy().reshape(-1, 1)
    y_test = df_test['egap_class'].to_numpy()
    fig, ax = plt.subplots()
    display = RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=ax)
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax.set_xlabel('False positive rate', fontsize=12)
    ax.set_ylabel('True positive rate', fontsize=12)
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/roc_curve.png', bbox_inches='tight', dpi=600)

    # print ROC-AUC score
    auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    print(f"ROC-AUC score on test set: {auc}")

    return df_test


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
        df = get_results_df(workdir)
        df.head()
        print(df)
        df.to_csv(df_path, index=False)
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df = pd.read_csv(df_path)

    df['abs. error'] = abs(df['prediction'] - df[config.label_str])
    # get dataframe with only split data
    df_train = df.loc[lambda df_temp: df_temp['split'] == 'train']
    mean_abs_err_train = df_train.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on train set: {mean_abs_err_train}')

    df_val = df.loc[lambda df_temp: df_temp['split'] == 'validation']
    mean_abs_err_val = df_val.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on validation set: {mean_abs_err_val}')

    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
    mean_abs_err_test = df_test.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on test set: {mean_abs_err_test}')

    mean_target = df.mean(0, numeric_only=True)[config.label_str]
    std_target = df.std(0, numeric_only=True)[config.label_str]
    print(f'Target mean: {mean_target}, std: {std_target} for {config.label_str}')

    # TODO: also filter half metals
    df_metal = df.loc[lambda df_temp: df_temp['Egap_type'] == 'metal']
    df_non_metal = df.loc[lambda df_temp: df_temp['Egap_type'] != 'metal']
    mean_abs_err = df_non_metal.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on test set, non_metals: {mean_abs_err}')

    #plot_egap_hist(df)
    df_classes = classify_egap(df, workdir)
    df_non_metals_predicted_test = df_classes.loc[
        lambda df_temp: df_temp['egap_class_predict'] == 1
    ]
    df_nm = df_non_metals_predicted_test
    mean_abs_err = df_nm['abs. error'].mean()
    print(f'MAE on test set, predicted non_metals: {mean_abs_err}')

    rmse = ((df_nm['abs. error'] ** 2).mean()) ** .5
    print(f'RMSE on test set, predicted non_metals: {rmse}')

    r2_test = 1 - (df_nm['abs. error'] ** 2).mean()/(df_nm['Egap'].std()) ** 2
    print(f'R^2 on test set, predicted non_metals: {r2_test}')


if __name__ == "__main__":
    app.run(main)

"""Combine result csv files in FLAGS.files, and do a combined analysis."""

import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats


FLAGS = flags.FLAGS
flags.DEFINE_list('paths', None, 'csv file paths', required=True)
flags.DEFINE_list('labels', None, 'labels for the prediction columns', required=True)
flags.DEFINE_string('out_path', 'results/aflow/result_combined.csv', 'output path')
flags.DEFINE_list('plots', ['all'], 'Which plots to plot.')
flags.DEFINE_string('index_col', 'auid', 'Column used to index the data')
flags.DEFINE_list('units', None, 'Units for each of the files')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the csv.')
flags.DEFINE_bool('redo', False, 'If the csv combine should be redone')
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')


def plot_ef_parity(df):
    """Plot the regression using joint plot with marginal histograms."""
    rows = (np.abs(stats.zscore(
        df[['enthalpy_formation_atom', 'ef_pred']],
        nan_policy='omit')) < 3).all(axis=1)
    g = sns.JointGrid(
        data=df[rows], x='enthalpy_formation_atom', y='ef_pred',
        marginal_ticks=False, height=5
    )

    # Add the joint and marginal histogram plots
    g.plot_joint(
        sns.histplot, discrete=(False, False), bins=(50, 50),
    )
    g.plot_marginals(sns.histplot, element="step", color=None)
    g.ax_marg_x.set_xlabel('Count')
    g.ax_marg_y.set_ylabel('Count')
    g.ax_joint.tick_params(which='both')
    g.ax_joint.set_xlabel(r'Calculated $E_f$')
    g.ax_joint.set_ylabel(r'Predicted $E_f$')
    #g.ax_joint.set_xticks([-4, -2, 0, 2, 4])
    #g.ax_joint.set_yticks([-4, -2, 0, 2, 4])
    x_ref = np.linspace(*g.ax_joint.get_xlim())
    g.ax_joint.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    #plt.xlabel(CALCULATE_LABEL, fontsize=FLAGS.font_size)
    #plt.xlabel(PREDICT_LABEL, fontsize=FLAGS.font_size)
    plt.tight_layout()
    plt.show()
    g.savefig('results/aflow/ef_parity.png', bbox_inches='tight', dpi=600)


def plot_egap_hist(df):
    """Plot a histogram of bandgap values."""
    rows = (np.abs(stats.zscore(
        df[['egap_pred']],
        nan_policy='omit')) < 3).all(axis=1)
    df_plot = df[rows]
    df_ins = df_plot[df_plot['class_pred'] == 1]
    fig, ax = plt.subplots(2)
    sns.histplot(
        df_plot,
        x='egap_pred',
        hue='Predicted class',
        ax=ax[0],
        log_scale=(False, False),
        multiple='stack'
    )
    ax[0].set_xlabel(r'Predicted $E_g$')
    sns.histplot(
        df_ins,
        x='egap_pred',
        ax=ax[1],
        log_scale=(False, False),
    )
    ax[1].set_xlabel(r'Predicted $E_g$')
    fig.savefig('results/aflow/egap_hist.png', bbox_inches='tight', dpi=600)


def main(_):
    """This is a docstring for the main function."""
    if FLAGS.units is None:
        FLAGS.units = ['a.u.']*len(FLAGS.paths)

    if (not os.path.exists(FLAGS.out_path)) or FLAGS.redo:
        logging.info(f"Did not find {FLAGS.out_path}, combining csv")
        df_all = pd.DataFrame({})
        for path, label in zip(FLAGS.paths, FLAGS.labels):
            logging.info(f'Reading {path}')
            if FLAGS.limit is None:
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path, nrows=FLAGS.limit)
            if 'numbers' in df.columns:
                df.drop(['numbers'], axis=1, inplace=True)
            df[label+'_pred'] = df['prediction']
            if 'prediction_std' in df:
                df[label+'_pred_std'] = df['prediction_std']
            df.index = df['auid']
            print(df.describe())
            print(df.columns)
            df_all = df_all.combine_first(df)
        df_all.drop(['prediction', 'prediction_std'], axis=1, inplace=True)
        print(df_all.describe())
        print(df_all.columns)
        logging.info(f"Wrote dataframe to {FLAGS.out_path}")
        df_all.to_csv(FLAGS.out_path, mode='w')
        #df_all = df_all.dropna()
    logging.info(f"Loading dataframe from {FLAGS.out_path}")
    df = pd.read_csv(FLAGS.out_path)

    plt.rc('xtick', labelsize=FLAGS.tick_size)
    plt.rc('ytick', labelsize=FLAGS.tick_size)
    plt.rc('legend', fontsize=FLAGS.font_size)
    plt.rc('legend', title_fontsize=FLAGS.font_size)
    plt.rc('axes', labelsize=FLAGS.font_size)

    df['p_insulator'] = 1 - df['egap_class_pred']
    # calculate the class prediction by applying a threshold. Because of the
    # softmax outputs probability, the threshold is exactly 1/2
    df['class_pred'] = df['p_insulator'].apply(lambda p: (p > 0.5)*1)
    df_ins = df[df['class_pred'] == 1]
    print(f"Number of entries predicted to be insulators: {len(df_ins)}")
    df['Predicted class'] = df['class_pred'].map(
        {0: 'metal', 1: 'non-metal'})

    if 'ef' in FLAGS.plots or FLAGS.plots[0] == 'all':
        plot_ef_parity(df)
    if 'egap' in FLAGS.plots or FLAGS.plots[0] == 'all':
        plot_egap_hist(df)


if __name__ == "__main__":
    app.run(main)

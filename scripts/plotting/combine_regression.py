"""This script produces plots to evaluate the errors of the model inferences,
using different metrics such as atomic numbers, number of species etc.
"""
from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('label', 'ef', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "ef" or "egap"')
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')


PREDICT_LABEL = ''
CALCULATE_LABEL = ''
ABS_ERROR_LABEL = ''


def plot_regression(df, workdir, label_str, plot_name):
    """Plot the regression using joint plot with marginal histograms."""
    if FLAGS.label == 'egap':
        xlim = [-0.5, 12.5]
        ylim = [-0.5, 12.5]
    elif FLAGS.label == 'U0':
        xlim = None
        ylim = None
    else:
        xlim = None
        ylim = None
    g = sns.JointGrid(
        data=df, x=label_str, y='prediction', marginal_ticks=False,
        height=5, xlim=xlim, ylim=ylim
    )

    # Add the joint and marginal histogram plots
    g.plot_joint(
        sns.histplot, discrete=(False, False), bins=(50, 50),
    )
    g.plot_marginals(sns.histplot, element="step", color=None)
    g.ax_marg_x.set_xlabel('Count', fontsize=FLAGS.font_size)
    g.ax_marg_y.set_ylabel('Count', fontsize=FLAGS.font_size)
    g.ax_joint.tick_params(which='both', labelsize=FLAGS.tick_size)
    g.ax_joint.set_xlabel(CALCULATE_LABEL, fontsize=FLAGS.font_size)
    g.ax_joint.set_ylabel(PREDICT_LABEL, fontsize=FLAGS.font_size)
    if FLAGS.label == 'egap':
        g.ax_joint.set_xticks([0, 2, 4, 6, 8, 10, 12])
        g.ax_joint.set_yticks([0, 2, 4, 6, 8, 10, 12])
    elif FLAGS.label == 'U0':
        pass
    else:
        g.ax_joint.set_xticks([-4, -2, 0, 2])
        g.ax_joint.set_yticks([-4, -2, 0, 2])
    x_ref = np.linspace(*g.ax_joint.get_xlim())
    g.ax_joint.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    #plt.xlabel(CALCULATE_LABEL, fontsize=FLAGS.font_size)
    #plt.xlabel(PREDICT_LABEL, fontsize=FLAGS.font_size)
    plt.tight_layout()
    plt.show()
    g.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)


def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # set correct axis labels
    global PREDICT_LABEL
    global CALCULATE_LABEL
    global ABS_ERROR_LABEL
    if FLAGS.label == 'egap':
        PREDICT_LABEL = r'Predicted $E_g$ (eV)'
        CALCULATE_LABEL = r'Calculated $E_g$ (eV)'
        ABS_ERROR_LABEL = 'Abs. error (eV)'
    elif FLAGS.label == 'U0':
        PREDICT_LABEL = r'Predicted $U_0$ (eV)'
        CALCULATE_LABEL = r'Calculated $U_0$ (eV)'
        ABS_ERROR_LABEL = 'Abs. error (eV)'
    else:
        PREDICT_LABEL = r'Predicted $E_f$ (eV/atom)'
        CALCULATE_LABEL = r'Calculated $E_f$ (eV/atom)'
        ABS_ERROR_LABEL = 'Abs. error (eV/atom)'

    df_aflow_to_aflow = pd.read_csv('results/aflow_x_mp/ef/train_aflow/result.csv')
    df_mp_to_mp = pd.read_csv('results/aflow_x_mp/ef/train_mp/result.csv')
    df_aflow_to_mp = pd.read_csv('results/aflow_x_mp/ef/infer_mp/result.csv')
    df_mp_to_aflow = pd.read_csv('results/aflow_x_mp/ef/infer_aflow/result.csv')

    df_aflow_to_aflow = df_aflow_to_aflow.loc[lambda df_temp: df_temp['split'] == 'test']
    df_mp_to_mp = df_mp_to_mp.loc[lambda df_temp: df_temp['split'] == 'test']
    df_aflow_to_mp = df_aflow_to_mp.loc[lambda df_temp: df_temp['split'] == 'test']
    df_mp_to_aflow = df_mp_to_aflow.loc[lambda df_temp: df_temp['split'] == 'test']

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
    sns.histplot(data=df_aflow_to_aflow, x='enthalpy_formation_atom', y='prediction',
        ax=ax[0, 0], bins=(50, 50))
    x_ref = np.linspace(*ax[0, 0].get_xlim())
    ax[0, 0].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[0, 0].set(
        ylabel=r'AFLOW model $E_f$ (eV/atom)',
        xlabel=r'AFLOW DFT $E_f$ (eV/atom)')
    ax[0, 0].text(0.05, 0.9, "MAE: 30 meV/atom", transform=ax[0, 0].transAxes)

    sns.histplot(data=df_mp_to_mp, x='delta_e', y='prediction',
        ax=ax[1, 1], bins=(50, 50))
    ax[1, 1].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[1, 1].set(
        ylabel=r'MP model $E_f$ (eV/atom)',
        xlabel=r'MP DFT $E_f$ (eV/atom)')
    ax[1, 1].text(0.05, 0.9, "MAE: 23 meV/atom", transform=ax[1, 1].transAxes)

    sns.histplot(data=df_aflow_to_mp, x='delta_e', y='prediction',
        ax=ax[0, 1], bins=(50, 50))
    ax[0, 1].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[0, 1].set(
        ylabel=r'AFLOW model $E_f$ (eV/atom)',
        xlabel=r'MP DFT $E_f$ (eV/atom)')
    ax[0, 1].text(0.05, 0.9, "MAE: 578 meV/atom", transform=ax[0, 1].transAxes)

    sns.histplot(data=df_mp_to_aflow, x='enthalpy_formation_atom', y='prediction',
        ax=ax[1, 0], bins=(50, 50))
    ax[1, 0].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[1, 0].set(
        ylabel=r'MP model $E_f$ (eV/atom)',
        xlabel=r'AFLOW DFT $E_f$ (eV/atom)')
    ax[1, 0].text(0.05, 0.9, "MAE: 238 meV/atom", transform=ax[1, 0].transAxes)

    for i in ax.flatten():
        i.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    fig.savefig('databases/combined_regression.png', bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    app.run(main)

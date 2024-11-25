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
flags.DEFINE_integer('font_size', 13, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 13, 'font size to use in labels')
flags.DEFINE_string('dir', 'results/aflow_x_mp/egap_all/', 'Directory \
                    that contains workdirs of the four training/eval runs.')
flags.DEFINE_string('label', 'ef', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "ef" or "egap"')

PREDICT_LABEL = ''
CALCULATE_LABEL = ''
ABS_ERROR_LABEL = ''


def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    plt.rcParams.update({'font.size': FLAGS.font_size})
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    match FLAGS.label:
        case 'ef':
            aflow_label_str = 'enthalpy_formation_atom'
            mp_label_str = 'delta_e'
            combi_label_str = 'ef_atom'
            unit_mae = 'meV/atom'
            unit_axis = 'eV/atom'
            mul = 1000 # multipier to from fit units to display units
            symbol = '$E_f$'
        case 'egap':
            aflow_label_str = 'Egap'
            mp_label_str = 'band_gap'
            combi_label_str = 'band_gap'
            unit_mae = 'meV'
            unit_axis = 'eV'
            mul = 1000 # multipier to from fit units to display units
            symbol = '$E_g$'

    # workdirs are named after training/testing combinations,
    # i.e. workdir with train on x and test on y is called 'x_infer_y'
    data_names = ['aflow', 'mp', 'combined']
    label_names = {
        'aflow': aflow_label_str,
        'mp': mp_label_str,
        'combined': combi_label_str
    }
    axis_label_dict = {
        'aflow': 'AFLOW',
        'mp': 'MP',
        'combined': 'Combined'
    }
    fig, ax = plt.subplots(
        len(data_names), len(data_names), sharex=True, sharey=True,
        figsize=(3*len(data_names), 3*len(data_names)))
    for i, train_name in enumerate(reversed(data_names)):
        for j, test_name in enumerate(data_names):
            df_dir = FLAGS.dir \
                + train_name + '_infer_' + test_name + '/result.csv'
            df = pd.read_csv(df_dir)
            df = df.loc[lambda df_temp: df_temp['split'] == 'test']
            label_str = label_names[test_name]
            residuals = df['prediction'] - df[label_str]
            mae = abs(residuals).mean()

            sns.histplot(data=df, x=label_str, y='prediction',
                ax=ax[i, j], bins=(50, 50))
            ax[i, j].set_xlim(-6, 4)
            ax[i, j].set_ylim(-6, 4)
            x_ref = np.linspace(*ax[i, j].get_xlim())
            ax[i, j].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
            ax[i, j].set(
                ylabel=fr'{axis_label_dict[train_name]} model {symbol} ({unit_axis})',
                xlabel=fr'{axis_label_dict[test_name]} target {symbol} ({unit_axis})')
            ax[i, j].text(0.05, 0.9, f"MAE: {mae*mul:.0f} {unit_mae}",
                transform=ax[i, j].transAxes)
            if FLAGS.label=='ef':
                ax[i, j].set_xticks([-6, -4, -2, 0, 2, 4])
                ax[i, j].set_yticks([-6, -4, -2, 0, 2, 4])
    for i in ax.flatten():
        i.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    fig.savefig(
        FLAGS.dir+f'combined_regression_{FLAGS.label}.png',
        bbox_inches='tight', dpi=600)
    exit()


    df_aflow_to_aflow = pd.read_csv(FLAGS.dir+'train_aflow/result.csv')
    df_mp_to_mp = pd.read_csv(FLAGS.dir+'train_mp/result.csv')
    df_aflow_to_mp = pd.read_csv(FLAGS.dir+'infer_mp/result.csv')
    df_mp_to_aflow = pd.read_csv(FLAGS.dir+'infer_aflow/result.csv')

    # filter to only include test split
    df_aflow_to_aflow = df_aflow_to_aflow.loc[lambda df_temp: df_temp['split'] == 'test']
    df_mp_to_mp = df_mp_to_mp.loc[lambda df_temp: df_temp['split'] == 'test']
    df_aflow_to_mp = df_aflow_to_mp.loc[lambda df_temp: df_temp['split'] == 'test']
    df_mp_to_aflow = df_mp_to_aflow.loc[lambda df_temp: df_temp['split'] == 'test']

    # calculate MAEs
    mae_af_to_af = df_aflow_to_aflow['prediction'] - df_aflow_to_aflow[aflow_label_str]
    mae_af_to_af = abs(mae_af_to_af).mean()
    mae_mp_to_mp = df_mp_to_mp['prediction'] - df_mp_to_mp[mp_label_str]
    mae_mp_to_mp = abs(mae_mp_to_mp).mean()
    mae_af_to_mp = df_aflow_to_mp['prediction'] - df_aflow_to_mp[mp_label_str]
    mae_af_to_mp = abs(mae_af_to_mp).mean()
    mae_mp_to_af = df_mp_to_aflow['prediction'] - df_mp_to_aflow[aflow_label_str]
    mae_mp_to_af = abs(mae_mp_to_af).mean()

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
    sns.histplot(data=df_aflow_to_aflow, x=aflow_label_str, y='prediction',
        ax=ax[0, 0], bins=(50, 50))
    x_ref = np.linspace(*ax[0, 0].get_xlim())
    ax[0, 0].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[0, 0].set(
        ylabel=fr'AFLOW model {symbol} ({unit_axis})',
        xlabel=fr'AFLOW target {symbol} ({unit_axis})')
    ax[0, 0].text(0.05, 0.9, f"MAE: {mae_af_to_af*mul:.0f} {unit_mae}",
        transform=ax[0, 0].transAxes)

    sns.histplot(data=df_mp_to_mp, x=mp_label_str, y='prediction',
        ax=ax[1, 1], bins=(50, 50))
    ax[1, 1].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[1, 1].set(
        ylabel=fr'MP model {symbol} ({unit_axis})',
        xlabel=fr'MP target {symbol} ({unit_axis})')
    ax[1, 1].text(0.05, 0.9, f"MAE: {mae_mp_to_mp*mul:.0f} {unit_mae}",
        transform=ax[1, 1].transAxes)

    sns.histplot(data=df_aflow_to_mp, x=mp_label_str, y='prediction',
        ax=ax[0, 1], bins=(50, 50))
    ax[0, 1].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[0, 1].set(
        ylabel=fr'AFLOW model {symbol} ({unit_axis})',
        xlabel=fr'MP target {symbol} ({unit_axis})')
    ax[0, 1].text(0.05, 0.9, f"MAE: {mae_af_to_mp*mul:.0f} {unit_mae}",
        transform=ax[0, 1].transAxes)

    sns.histplot(data=df_mp_to_aflow, x=aflow_label_str, y='prediction',
        ax=ax[1, 0], bins=(50, 50))
    ax[1, 0].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[1, 0].set(
        ylabel=fr'MP model {symbol} ({unit_axis})',
        xlabel=fr'AFLOW target {symbol} ({unit_axis})')
    ax[1, 0].text(0.05, 0.9, f"MAE: {mae_mp_to_af*mul:.0f} {unit_mae}",
        transform=ax[1, 0].transAxes)

    for i in ax.flatten():
        i.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    fig.savefig(
        FLAGS.dir+'combined_regression.png',
        bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    app.run(main)

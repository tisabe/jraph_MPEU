"""This script produces plots to evaluate the errors of the model inferences,
using different metrics such as atomic numbers, number of species etc.
"""
import os
from collections import Counter

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn.metrics

from jraph_MPEU.utils import load_config, str_to_list
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')
flags.DEFINE_string('label', 'ef', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "ef" or "egap"')
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')

PREDICT_LABEL = ''
CALCULATE_LABEL = ''
ABS_ERROR_LABEL = ''


def plot_error_hist(df, workdir, plot_name, hue_val):
    """Plot absolute errors in a log-log histogram."""
    fig, ax = plt.subplots()
    sns.histplot(
        x='abs. error', hue=hue_val, data=df, ax=ax, multiple='stack',
        palette='Paired', log_scale=True)
    ax.set_xlabel(ABS_ERROR_LABEL, fontsize=FLAGS.font_size)
    ax.set_ylabel('Count', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    #ax.legend(title='Bandgap type')
    plt.rc('legend', fontsize=FLAGS.tick_size-3)
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)


def plot_regression(df, workdir, label_str, plot_name):
    """Plot the regression using joint plot with marginal histograms."""
    if FLAGS.label == 'egap':
        xlim = [-0.5, 12.5]
        ylim = [-0.5, 12.5]
    elif FLAGS.label == 'energy':
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
    elif FLAGS.label == 'energy':
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


def plot_regression_oxides(df, workdir, config, plot_name):
    """Plot the regression results for only oxide materials."""
    df_copy = df.copy()  # dataframe will be modified, so copy it before
    # filter dataframe rows for oxides
    df_copy = df_copy[df_copy['formula'].map(lambda formula: 'Fe' in formula)]

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=config.label_str, y='prediction', hue='split', data=df_copy, ax=ax)
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)


def plot_dft_type(df, workdir, plot_name):
    fig, ax = plt.subplots()
    sns.boxplot(
        x='dft_type', # plot error vs dft type
        y='abs. error',
        data=df,
        ax=ax,
        hue='split'
    )
    plt.legend([], [], frameon=False)
    plt.xticks(rotation=90)
    ax.set_xlabel('AFLOW DFT type label', fontsize=FLAGS.font_size)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)


def plot_space_groups(df, workdir, plot_name, counts):
    fig, ax = plt.subplots()
    sns.boxplot(
        x='crystal system', # plot error vs space group
        y='abs. error',
        data=df,
        #hue='split',
        ax=ax,
        color='lightblue'
    )
    plt.legend([], [], frameon=False)
    plt.axhline(y=df['abs. error'].median(), alpha=0.8, color='red', linestyle='--')
    plt.xticks(rotation=90)
    ax.set_xlabel('', fontsize=FLAGS.font_size)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    # write counts at the top of the plot
    bottom, top = ax.get_ylim()
    #ax.set_ylim(top=top*1)
    for xpos, xlabel in zip(ax.get_xticks(), ax.get_xticklabels()):
        #print(xtick)
        ax.text(
            xpos, top*0.8, counts[xlabel.get_text()],
            horizontalalignment='center', fontsize=FLAGS.font_size*0.8,
            bbox=dict(boxstyle="square", ec='black', fc='white'))
    plt.yscale('log')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)


def plot_bandgap_type(df, workdir, plot_name):
    fig, ax = plt.subplots()
    sns.boxplot(
        x='Egap_type', # plot error vs bandgap type
        y='abs. error',
        data=df,
        ax=ax,
        color='deepskyblue'
    )
    plt.axhline(y=df['abs. error'].median(), alpha=0.8, color='red', linestyle='--')
    plt.xticks(rotation=90)
    ax.set_xlabel('AFLOW band gap-type label', fontsize=FLAGS.font_size)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)


def plot_density(df, workdir, plot_name):
    fig, ax = plt.subplots()
    sns.histplot(
        x='density', # plot error vs density
        y='abs. error',
        data=df,
        ax=ax,
        cbar=True, cbar_kws={'label': 'Count'},
        log_scale=True
    )
    ax.set_xlabel(r'Density $(g/cm^3)$', fontsize=FLAGS.font_size)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=FLAGS.font_size)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)


def plot_ldau(df, workdir, plot_name):
    fig, ax = plt.subplots()
    sns.boxplot(
        x='ldau_type', # plot error vs ldau type
        y='abs. error',
        data=df,
        color='deepskyblue',
        ax=ax
    )
    ax.set_xlabel('AFLOW LDAU-type label', fontsize=FLAGS.font_size)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    plt.axhline(y=df['abs. error'].median(), alpha=0.8, color='grey', linestyle='--')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)


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
    elif FLAGS.label == 'energy':
        PREDICT_LABEL = r'Predicted $U_0$ (eV)'
        CALCULATE_LABEL = r'Calculated $U_0$ (eV)'
        ABS_ERROR_LABEL = 'Abs. error (eV)'
    else:
        PREDICT_LABEL = r'Predicted $E_f$ (eV/atom)'
        CALCULATE_LABEL = r'Calculated $E_f$ (eV/atom)'
        ABS_ERROR_LABEL = 'Abs. error (eV/atom)'
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
    if not 'prediction' in df.columns:
        df['prediction'] = df['prediction_mean']
    df['abs. error'] = abs(df['prediction'] - df[config.label_str])
    df['num_atoms'] = df['numbers'].apply(len)
    df['num_species'] = df['numbers'].apply(lambda num_list: len(set(num_list)))

    # group the spacegoups into crystal systems
    bins = [0, 2, 15, 74, 142, 167, 194, 230]
    labels = [
        'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal',
        'Trigonal', 'Hexagonal', 'Cubic']
    if 'spacegroup_relax' in df.columns:
        df['crystal system'] = pd.cut(df['spacegroup_relax'], bins, labels=labels)
    else:
        print('Skipping spacegroup conversion.')

    if 'Egap_type' in df.columns:
        df['Egap_type'] = df['Egap_type'].apply(lambda gap: gap.replace('_spin-polarized', ''))
    else:
        print("Egap_type not found in properties, continuing without.")
    if 'dft_type' in df.columns:
        df['dft_type'] = df['dft_type'].apply(lambda dft: dft.strip(" '[]"))
    else:
        print("dft_type not found in properties, continuing without.")

    # get dataframe with only split data
    df_train = df.loc[lambda df_temp: df_temp['split'] == 'train']
    mean_abs_err_train = df_train.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on train set: {mean_abs_err_train}')
    rmse_train = (df_train['abs. error'] ** 2).mean() ** .5
    print(f'RMSE on train set: {rmse_train}')

    df_val = df.loc[lambda df_temp: df_temp['split'] == 'validation']
    mean_abs_err_val = df_val.mean(0, numeric_only=True)['abs. error']
    rmse_val = (df_val['abs. error'] ** 2).mean() ** .5
    print(f'MAE on validation set: {mean_abs_err_val}')
    print(f'RMSE on validation set: {rmse_val}')

    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
    mean_abs_err_test = df_test.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on test set: {mean_abs_err_test}')
    rmse_test = (df_test['abs. error'] ** 2).mean() ** .5
    print(f'RMSE on test set: {rmse_test}')
    stdev = np.std(df_test[config.label_str])
    print(f'STDEV of test set: {stdev}')
    r2_test = sklearn.metrics.r2_score(
        df_test[config.label_str], df_test['prediction'])
    print(f'R^2 on test set: {r2_test}')
    median_err = df_test.median(0, numeric_only=True)['abs. error']
    print(f'Median error on test set: {median_err}')

    # print rows with highest errors
    df_test = df_test.sort_values(by='abs. error', axis='index')
    print(df_test[-3:][['auid', 'prediction', config.label_str, 'abs. error',
        'formula', 'crystal system', 'Egap']])
    """
    row_min_err = df_test.loc[df_test['abs. error'].idxmin()]
    print(row_min_err)

    mean_target = df.mean(0, numeric_only=True)[config.label_str]
    std_target = df.std(0, numeric_only=True)[config.label_str]
    print(f'Target mean: {mean_target}, std: {std_target} for {config.label_str}')
    
    fig, ax = plt.subplots()
    sns.histplot(
        x=config.label_str, y='prediction', data=df_test, ax=ax,
        cbar=True, cbar_kws={'label': 'Count'}, bins=(100, 100))
    x_ref = np.linspace(*ax.get_xlim())
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    ax.set_xlabel(CALCULATE_LABEL, fontsize=FLAGS.font_size)
    ax.set_ylabel(PREDICT_LABEL, fontsize=FLAGS.font_size)
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/hist_simple.png', bbox_inches='tight', dpi=600)

    #plot_regression_oxides(df_test, workdir, config, '/regression_oxides.png')

    fig, ax = plt.subplots()
    sns.boxplot(
        x='num_species',
        y='abs. error',
        data=df_test,
        color='deepskyblue',
        ax=ax,
    )
    plt.legend([], [], frameon=False)
    ax.set_xlabel('Number of species in compound', fontsize=FLAGS.font_size)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/error_vs_nspecies.png', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    sns.histplot(
        x='num_atoms',
        y='abs. error',
        data=df_test,
        ax=ax,
        cbar=True, cbar_kws={'label': 'Count'},
        log_scale=(False, True),
        bins=max(df_test['num_atoms'])
    )
    ax.set_xlabel('Number of atoms in unit cell', fontsize=FLAGS.font_size)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/error_vs_natoms.png', bbox_inches='tight', dpi=600)
    """
    plot_regression(df_test, workdir, config.label_str, '/regression_test.png')

    if 'spacegroup_relax' in df.columns:
        col = df_train['crystal system']
        counts = dict(Counter(col))
        print(counts)
        plot_space_groups(df_test, workdir, '/error_vs_crystal.png', counts)
        plt.pie(counts.values(), labels=counts.keys())
        plt.show()
    else:
        print('Skipping spacegroup plots.')

    if 'Egap_type' in df.columns:
        plot_error_hist(df_test, workdir, '/error_hist.png', 'Egap_type')
        col = df_train['Egap_type']
        print(Counter(col))
        plot_bandgap_type(df, workdir, '/error_vs_egap_type.png')
    else:
        plot_error_hist(df_test, workdir, '/error_hist.png', None)

    if 'density' in df.columns:
        plot_density(df_test, workdir, '/error_vs_density.png')

    if 'ldau_type' in df.columns:
        col = df_train['ldau_type']
        print(Counter(col))
        plot_ldau(df, workdir, '/error_vs_ldau.png')


if __name__ == "__main__":
    app.run(main)

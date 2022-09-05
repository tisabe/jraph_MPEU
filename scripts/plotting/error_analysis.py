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

from jraph_MPEU.utils import load_config, str_to_list
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')

PREDICT_LABEL = 'Predicted band gap (eV)'
CALCULATE_LABEL = 'Calculated band gap (eV)'
ABS_ERROR_LABEL = 'MAE (eV)'

def plot_regression(df, workdir, config, plot_name, color=u'#1f77b4'):
    fig, ax = plt.subplots()
    sns.histplot(
        x=config.label_str,  # plot prediction vs label
        y='prediction',
        data=df,
        #hue='split',
        cbar=True, cbar_kws={'label': 'Count'},
        ax=ax,
        color=color
    )
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax.set_xlabel(CALCULATE_LABEL, fontsize=12)
    ax.set_ylabel(PREDICT_LABEL, fontsize=12)
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
    ax.set_xlabel('AFLOW DFT type label', fontsize=12)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)


def plot_space_groups(df, workdir, plot_name):
    # group the spacegoups into crystal systems
    #df['spacegroup_relax'] = df['spacegroup_relax'].astype('category')
    bins = [0, 2, 15, 74, 142, 167, 194, 230]
    labels = [
        'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal',
        'Trigonal', 'Hexagonal', 'Cubic']
    df['crystal system'] = pd.cut(df['spacegroup_relax'], bins, labels=labels)
    fig, ax = plt.subplots()
    sns.boxplot(
        x='crystal system', # plot error vs space group
        y='abs. error',
        data=df,
        hue='split',
        ax=ax
    )
    plt.legend([], [], frameon=False)
    plt.axhline(y=df['abs. error'].median(), alpha=0.8, color='grey', linestyle='--')
    plt.xticks(rotation=90)
    ax.set_xlabel('Crystal system', fontsize=12)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+plot_name, bbox_inches='tight', dpi=600)

    col = df['crystal system']
    print(Counter(col))


def plot_bandgap_type(df, workdir, plot_name):
    fig, ax = plt.subplots()
    sns.boxplot(
        x='Egap_type', # plot error vs bandgap type
        y='abs. error',
        data=df,
        ax=ax,
        hue='split'
    )
    plt.axhline(y=df['abs. error'].median(), alpha=0.8, color='grey', linestyle='--')
    plt.xticks(rotation=90)
    ax.set_xlabel('AFLOW band gap-type label', fontsize=12)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=12)
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
    ax.set_xlabel(r'Density $(g/cm^3)$', fontsize=12)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=12)
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
        hue='split',
        ax=ax
    )
    ax.set_xlabel('AFLOW LDAU-type label', fontsize=12)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=12)
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
        df['numbers'] = df['numbers'].apply(str_to_list)

    df['abs. error'] = abs(df['prediction'] - df[config.label_str])
    df['num_atoms'] = df['numbers'].apply(len)
    df['num_species'] = df['numbers'].apply(lambda num_list: len(set(num_list)))
    try:
        df['Egap_type'] = df['Egap_type'].apply(lambda gap: gap.replace('_spin-polarized', ''))
    except KeyError:
        print("Egap_type not found in properties, continuing without.")
    try:
        df['dft_type'] = df['dft_type'].apply(lambda dft: dft.strip(" '[]"))
    except KeyError:
        print("dft_type not found in properties, continuing without.")
    # get dataframe with only split data
    df_train = df.loc[lambda df_temp: df_temp['split'] == 'train']
    mean_abs_err_train = df_train.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on train set: {mean_abs_err_train}')

    df_val = df.loc[lambda df_temp: df_temp['split'] == 'validation']
    mean_abs_err_val = df_val.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on validation set: {mean_abs_err_val}')

    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
    # exclude outliers
    df_test_filter = df_test[df_test[config.label_str] < 50.0]
    mean_abs_err_test = df_test_filter.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on test set: {mean_abs_err_test}')
    rmse_test = (df_test_filter['abs. error'] ** 2).mean() ** .5
    print(f'RMSE on test set: {rmse_test}')
    r2_test = 1 - (df_test_filter['abs. error'] ** 2).mean()/df_test_filter[
            config.label_str].std()
    print(f'R^2 on test set: {r2_test}')

    mean_target = df.mean(0, numeric_only=True)[config.label_str]
    std_target = df.std(0, numeric_only=True)[config.label_str]
    print(f'Target mean: {mean_target}, std: {std_target} for {config.label_str}')

    sns.scatterplot(x=config.label_str, y='prediction', data=df_test)
    plt.show()

    fig, ax = plt.subplots()
    sns.boxplot(
        x='num_species',
        y='abs. error',
        data=df_test,
        hue='split',
        ax=ax,
    )
    plt.legend([], [], frameon=False)
    ax.set_xlabel('Number of species in compound', fontsize=12)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=12)
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
    ax.set_xlabel('Number of atoms in unit cell', fontsize=12)
    ax.set_ylabel(ABS_ERROR_LABEL, fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/error_vs_natoms.png', bbox_inches='tight', dpi=600)

    plot_regression(df_test, workdir, config, '/regression_test.png', color=u'#1f77b4')
    #plot_regression(df_train, workdir, config, '/regression_train.png', color=u'#2ca02c')
    #plot_regression(df_val, workdir, config, '/regression_val.png', color=u'#ff7f0e')

    plot_dft_type(df_test, workdir, '/dft_type_error.png')
    col = df['dft_type']
    print(Counter(col))

    plot_space_groups(df_test, workdir, '/error_vs_crystal.png')

    plot_bandgap_type(df, workdir, '/error_vs_egap_type.png')
    col = df['Egap_type']
    print(Counter(col))

    plot_density(df_test, workdir, '/error_vs_density.png')

    plot_ldau(df, workdir, '/error_vs_ldau.png')


if __name__ == "__main__":
    app.run(main)

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
    # get dataframe with only test data
    df_test = df.loc[lambda df_temp: df_temp['split']=='test']

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=config.label_str,  # plot prediction vs label
        y='prediction',
        data=df,
        hue='split',
        ax=ax
    )
    plt.show()
    fig.savefig(workdir+'/pred_vs_label.png', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=config.label_str, # plot error vs label
        y='abs. error',
        data=df,
        hue='split',
        ax=ax
    )
    plt.show()
    fig.savefig(workdir+'/error_vs_label.png', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    sns.boxplot(
        x='dft_type', # plot error vs dft type
        y='abs. error',
        data=df,
        hue='split',
        ax=ax
    )
    plt.yscale('log')
    plt.show()
    fig.savefig(workdir+'/error_vs_label.png', bbox_inches='tight', dpi=600)
    
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
    plt.axhline(y=1e-2, color='black', linestyle='--')
    plt.yscale('log')
    plt.show()
    fig.savefig(workdir+'/error_vs_label.png', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    sns.boxplot(
        x='Egap_type', # plot error vs Egap type
        y='abs. error',
        data=df,
        hue='split',
        ax=ax
    )
    plt.axhline(y=1e-2, color='black', linestyle='--')
    plt.yscale('log')
    plt.show()
    fig.savefig(workdir+'/error_vs_label.png', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x='density', # plot error vs density
        y='abs. error',
        data=df,
        hue='split',
        ax=ax
    )
    plt.yscale('log')
    plt.show()
    fig.savefig(workdir+'/error_vs_label.png', bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    app.run(main)

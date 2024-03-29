"""This script plots and generates monte-carlo dropout inferences."""
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

from jraph_MPEU.utils import load_config
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/test_dropout', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')
flags.DEFINE_string('unit', 'eV/atom', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "ef" or "egap"')

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
        df = get_results_df(workdir, limit=FLAGS.limit, mc_dropout=True)
        df.head()
        print(df)
        df.to_csv(df_path, index=False)
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df = pd.read_csv(df_path)

    if not 'prediction_mean' in df.columns:
        df['prediction_mean'] = df['prediction']
    df['abs. error'] = abs(df['prediction_mean'] - df[config.label_str])
    bins = [0, 2, 15, 74, 142, 167, 194, 230]
    labels = [
        'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal',
        'Trigonal', 'Hexagonal', 'Cubic']
    df['crystal system'] = pd.cut(df['spacegroup_relax'], bins, labels=labels)

    # get dataframe with only split data
    df_train = df.loc[lambda df_temp: df_temp['split'] == 'train']
    mean_abs_err_train = df_train.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on train set: {mean_abs_err_train}')

    df_train_test = df.loc[lambda df_temp: df_temp['split'] != 'validation']

    df_val = df.loc[lambda df_temp: df_temp['split'] == 'validation']
    mean_abs_err_val = df_val.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on validation set: {mean_abs_err_val}')

    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
    mean_abs_err_test = df_test.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on test set: {mean_abs_err_test}')

    mean_target = df.mean(0, numeric_only=True)[config.label_str]
    std_target = df.std(0, numeric_only=True)[config.label_str]
    print(f'Target mean: {mean_target}, std: {std_target} for {config.label_str}')

    for split in ['train', 'validation', 'test']:
        df_split = df.loc[lambda df_temp: df_temp['split'] == split]
        r2_split = sklearn.metrics.r2_score(
            df_split['abs. error'], df_split['prediction_std']
        )
        print(f'Uncertainty R^2 on {split} set: {r2_split}')
        pearson = df_split[['abs. error', 'prediction_std']].corr()
        print(f'Pearson r^2 on {split} set: {pearson}')

    # convert split labels for dataframe with all splits
    split_convert = {
        'train': 'Training', 'validation': 'Validation', 'test': 'Test'}
    df_train_test['split'] = df_train_test['split'].apply(
        lambda x: split_convert[x])

    sns.set_context('paper')
    fig, ax = plt.subplots()
    sns.scatterplot(
        ax=ax,
        x='abs. error',
        y='prediction_std',
        data=df,
        hue='split'
    )
    ax.set_xlabel(f'Absolute error ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.set_ylabel(f'Prediction STDEV ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    #ax.legend(title='', fontsize=FLAGS.font_size-3)  # disable 'split' title
    ax.legend(loc=2, prop={'size': FLAGS.tick_size})
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/mc_error', bbox_inches='tight', dpi=600)
    corr = df_test['abs. error'].corr(df_test['prediction_std'])
    print(f"Correlation on test split: {corr}")

    fig, ax = plt.subplots()
    sns.violinplot(
        ax=ax,
        x='crystal system',
        y='prediction_std',
        data=df_test,
        linewidth=1,
        color='lightblue',
        inner='quartile',
        #cut=0,  # limit the violins to data range
    )
    ax.set_ylabel(f'Prediction STDEV ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size, width=0)
    ax.set_xlabel('')
    #ax.set_ylim(bottom=0, top=0.04)
    ax.set_ylim(bottom=0, top=0.2)

    col = df_train['crystal system']
    counts = Counter(col)
    bottom, top = ax.get_ylim()
    #ax.set_ylim(top=top*0.5)
    for xpos, xlabel in zip(ax.get_xticks(), ax.get_xticklabels()):
        #print(xtick)
        ax.text(
            xpos, top*0.9, counts[xlabel.get_text()],
            horizontalalignment='center', fontsize=FLAGS.font_size*0.8,
            bbox=dict(boxstyle="square", ec='black', fc='white'))
    #plt.yscale('log')
    plt.axhline(
        y=df_test['prediction_std'].median(), alpha=0.8, color='red', linestyle='--')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/mc_crystal_system', bbox_inches='tight', dpi=600)
    # print counts of different crystal systems
    col = df_train['crystal system']
    print(Counter(col))
    print(f"Minimum std: {df_test['prediction_std'].min()}")

    ### Plot DFT+U correction results

    # convert integer dft_type to string type
    int_to_string_type = {0: 'PBE', 2: 'PBE+U'}
    df_test['ldau_type'] = df_test['ldau_type'].apply(
        lambda x: int_to_string_type[x])

    fig, ax = plt.subplots()
    sns.violinplot(
        ax=ax,
        x='ldau_type',
        y='prediction_std',
        data=df_test,
        linewidth=1,
        color='lightblue',
        inner='quartile'
        #cut=0,  # limit the violins to data range
    )
    ax.set_ylabel(f'Prediction STDEV ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size, width=0)
    ax.set_xlabel('')
    #ax.set_ylim(bottom=0, top=0.04)
    ax.set_ylim(bottom=0, top=0.2)

    col = df_train['ldau_type']
    counts = Counter(col)
    ax.text(0.25, 0.90, counts[0.0], horizontalalignment='center', fontsize=FLAGS.font_size*0.8,
            bbox=dict(boxstyle="square", ec='black', fc='white'), transform=ax.transAxes)
    ax.text(0.75, 0.90, counts[2.0], horizontalalignment='center', fontsize=FLAGS.font_size*0.8,
            bbox=dict(boxstyle="square", ec='black', fc='white'), transform=ax.transAxes)
    plt.axhline(
        y=df_test['prediction_std'].median(), alpha=0.8, color='red', linestyle='--')
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/mc_ldau_type', bbox_inches='tight', dpi=600)
    # print counts of different dft types
    col = df_train['ldau_type']
    print(Counter(col))




if __name__ == "__main__":
    app.run(main)
